"""Forwarding NDN simplificado con FIB por prefijos, PIT y Content Store."""

from __future__ import annotations

from dataclasses import dataclass, field


class NoRouteError(RuntimeError):
    """El nodo no tiene ruta de prefijo para el Interest."""

# FibEntry representa una entrada en la FIB de un Forwarder, con un prefijo y un siguiente salto.
@dataclass(frozen=True)
class FibEntry:
    prefix: str
    next_hop: str

    def __post_init__(self) -> None:
        if not self.prefix.startswith("/") or self.prefix.endswith("/"):
            raise ValueError(f"Prefijo NDN no valido: {self.prefix}")

#Forwarder representa un nodo en la red NDN con una FIB, un Content Store y un PIT.
@dataclass
class Forwarder:
    name: str
    fib: list[FibEntry] = field(default_factory=list)
    content_store: set[str] = field(default_factory=set)
    pit: dict[str, str | None] = field(default_factory=dict)

    # lookup busca la entrada de FIB más larga que coincida con el nombre del Interest.
    def lookup(self, interest_name: str) -> FibEntry | None:
        matches = [
            entry
            for entry in self.fib
            if interest_name == entry.prefix
            or interest_name.startswith(entry.prefix + "/")
        ]
        return max(matches, key=lambda entry: len(entry.prefix)) if matches else None


@dataclass(frozen=True)
class ForwardingResult:
    interest_name: str
    interest_path: tuple[str, ...]
    data_path: tuple[str, ...]
    satisfied_by: str
    cache_hit: bool
    fib_lookups: tuple[tuple[str, str, str], ...]


class Network:
    def __init__(self, forwarders: dict[str, Forwarder]) -> None:
        self.forwarders = forwarders
        for forwarder in forwarders.values():
            for entry in forwarder.fib:
                if entry.next_hop not in forwarders:
                    raise ValueError(
                        f"Siguiente salto desconocido {entry.next_hop} en {forwarder.name}"
                    )

    def forward_interest(self, source: str, interest_name: str) -> ForwardingResult:
        if source not in self.forwarders:
            raise KeyError(f"Nodo de origen desconocido: {source}")
        if not interest_name.startswith("/") or interest_name.endswith("/"):
            raise ValueError(f"Nombre de Interest no valido: {interest_name}")

        current = source
        previous: str | None = None
        interest_path: list[str] = []
        fib_lookups: list[tuple[str, str, str]] = []
        visited: set[str] = set()

        while True:
            if current in visited:
                self._clear_pending(interest_name, interest_path)
                raise NoRouteError(f"Bucle de forwarding para {interest_name}")
            visited.add(current)
            forwarder = self.forwarders[current]
            interest_path.append(current)

            if interest_name in forwarder.content_store:
                satisfied_by = current
                cache_hit = current != source
                break

            entry = forwarder.lookup(interest_name)
            if entry is None:
                self._clear_pending(interest_name, interest_path[:-1])
                raise NoRouteError(
                    f"Sin ruta para {interest_name} en el nodo {current}"
                )

            forwarder.pit[interest_name] = previous
            fib_lookups.append((current, entry.prefix, entry.next_hop))
            previous, current = current, entry.next_hop

        data_path = tuple(reversed(interest_path[:-1]))
        for node_name in interest_path[:-1]:
            node = self.forwarders[node_name]
            node.pit.pop(interest_name, None)
            node.content_store.add(interest_name)

        return ForwardingResult(
            interest_name=interest_name,
            interest_path=tuple(interest_path),
            data_path=data_path,
            satisfied_by=satisfied_by,
            cache_hit=cache_hit,
            fib_lookups=tuple(fib_lookups),
        )

    def _clear_pending(self, interest_name: str, path: list[str]) -> None:
        for node_name in path:
            self.forwarders[node_name].pit.pop(interest_name, None)