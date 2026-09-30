import unittest

import numpy as np

from ndn_sim.experiment import infer_interest_from_context
from ndn_sim.network import FibEntry, Forwarder, Network, NoRouteError


class NetworkForwardingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.interest_name = "/ndn/video/canal1/segment/001"
        self.network = Network(
            {
                "edge": Forwarder("edge", [FibEntry("/ndn", "core")]),
                "core": Forwarder(
                    "core",
                    [
                        FibEntry("/ndn/video", "video"),
                        FibEntry("/ndn/video/canal1", "channel1"),
                    ],
                ),
                "video": Forwarder("video", content_store={self.interest_name}),
                "channel1": Forwarder(
                    "channel1", content_store={self.interest_name}
                ),
            }
        )

    def test_uses_longest_prefix_route_not_full_interest_route(self) -> None:
        result = self.network.forward_interest("edge", self.interest_name)

        self.assertEqual(result.interest_path, ("edge", "core", "channel1"))
        self.assertIn(("core", "/ndn/video/canal1", "channel1"), result.fib_lookups)
        self.assertTrue(all(prefix != self.interest_name for _, prefix, _ in result.fib_lookups))

    def test_data_returns_and_clears_pending_interests(self) -> None:
        result = self.network.forward_interest("edge", self.interest_name)

        self.assertEqual(result.data_path, ("core", "edge"))
        self.assertEqual(self.network.forwarders["edge"].pit, {})
        self.assertEqual(self.network.forwarders["core"].pit, {})

    def test_missing_prefix_route_fails(self) -> None:
        with self.assertRaises(NoRouteError):
            self.network.forward_interest("edge", "/outside/name")


class ContextInferenceTests(unittest.TestCase):
    def test_forwards_candidate_above_similarity_threshold(self) -> None:
        vocabulary = ["/ndn/context", "/ndn/candidate"]
        input_vectors = np.asarray([[1.0, 0.0], [0.8, 0.6]])

        prediction, score = infer_interest_from_context(
            ("/ndn/context",), vocabulary, input_vectors
        )

        self.assertEqual(prediction, "/ndn/candidate")
        self.assertGreaterEqual(score, 0.30)

    def test_rejects_candidate_below_similarity_threshold(self) -> None:
        vocabulary = ["/ndn/context", "/ndn/candidate"]
        input_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]])

        prediction, score = infer_interest_from_context(
            ("/ndn/context",), vocabulary, input_vectors
        )

        self.assertIsNone(prediction)
        self.assertLess(score, 0.30)


if __name__ == "__main__":
    unittest.main()