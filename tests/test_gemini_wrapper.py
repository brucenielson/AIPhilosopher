# tests/test_gemini_wrapper.py
import unittest
from unittest.mock import MagicMock, patch
from typing import Any

import models.gemini_wrapper as gw


class MockStrictContentType:
    def __init__(self, payload: Any):
        # keep the underlying payload so assertions can inspect it
        self.payload = payload

    def __repr__(self):
        return f"MockStrictContentType({self.payload!r})"


class DummyResourceExhausted(Exception):
    """Simple dummy exception to simulate google.api_core.exceptions.ResourceExhausted"""

    def __init__(self, details: str = ""):
        super().__init__("ResourceExhausted")
        self.details = details


class TestNormalizeHistory(unittest.TestCase):
    def setUp(self):
        # Patch content_types.to_content and StrictContentType used inside the module
        self._orig_ct = gw.content_types
        gw.content_types.to_content = lambda x: MockStrictContentType(x)
        gw.content_types.StrictContentType = MockStrictContentType

    def tearDown(self):
        # restore original reference to avoid interfering with other tests
        gw.content_types = self._orig_ct

    def test_none_history_returns_empty(self):
        normalized = gw.normalize_history_to_gemini_content(None)
        self.assertEqual(normalized, [])

    def test_list_of_lists_becomes_flat_strict_content(self):
        history = [["user1", "bot1"], ["user2", "bot2"]]
        normalized = gw.normalize_history_to_gemini_content(history)
        # should be 4 StrictContentType items, order preserved
        self.assertEqual(len(normalized), 4)
        self.assertTrue(all(isinstance(i, MockStrictContentType) for i in normalized))
        self.assertEqual([n.payload for n in normalized], ["user1", "bot1", "user2", "bot2"])

    def test_iterable_strings_and_dicts(self):
        history = ["hello", {"text": "from-dict"}]
        normalized = gw.normalize_history_to_gemini_content(history)
        self.assertEqual(len(normalized), 2)
        self.assertEqual([n.payload for n in normalized], ["hello", {"text": "from-dict"}])

    def test_unsupported_type_raises(self):
        history = [123]  # unsupported
        with self.assertRaises(TypeError):
            _ = gw.normalize_history_to_gemini_content(history)

    def test_malformed_inner_list_length_raises(self):
        # list-of-lists case requires inner lists of exactly length 2
        history = [["only-one-element"], ["a", "b"]]
        with self.assertRaises(ValueError):
            _ = gw.normalize_history_to_gemini_content(history)


class TestGeminiChatSessionWrapper(unittest.TestCase):
    def setUp(self):
        # patch content_types helpers
        self._orig_ct = gw.content_types
        gw.content_types.to_content = lambda x: MockStrictContentType(x)
        gw.content_types.StrictContentType = MockStrictContentType

    def tearDown(self):
        gw.content_types = self._orig_ct

    def test_send_message_forwards_and_get_history(self):
        chat = MagicMock()
        # set history attribute to something list-like
        chat.history = ["h1", "h2"]
        chat.send_message.return_value = "OK"
        wrapper = gw.GeminiChatSessionWrapper(chat_session=chat, history=["init"])
        result = wrapper.send_message("hi")
        self.assertEqual(result, "OK")
        # ensure the underlying send_message was called with a StrictContentType instance
        called_arg = chat.send_message.call_args.args[0]
        self.assertIsInstance(called_arg, MockStrictContentType)
        # get_history should return the chat history list
        self.assertEqual(wrapper.get_history(), ["h1", "h2"])

    def test_send_message_retries_on_resource_exhausted_then_succeeds(self):
        chat = MagicMock()
        chat.history = []
        # make chat.send_message raise a ResourceExhausted first, then return "OK"
        # patch the module's ResourceExhausted to our dummy class so the retry logic recognizes it
        with patch.object(gw, "ResourceExhausted", DummyResourceExhausted):
            # first call: raise ResourceExhausted, second call: "OK"
            chat.send_message.side_effect = [DummyResourceExhausted("retry_delay { seconds: 0 }"), "OK"]
            wrapper = gw.GeminiChatSessionWrapper(chat_session=chat)
            # patch time.sleep so test runs fast
            with patch("time.sleep", return_value=None):
                result = wrapper.send_message("hi")
        self.assertEqual(result, "OK")
        self.assertEqual(chat.send_message.call_count, 2)


class TestGeminiWrapper(unittest.TestCase):
    def setUp(self):
        # patch content_types used in normalization
        self._orig_ct = gw.content_types
        gw.content_types.to_content = lambda x: MockStrictContentType(x)
        gw.content_types.StrictContentType = MockStrictContentType

    def tearDown(self):
        gw.content_types = self._orig_ct

    def test_generate_content_forwards_to_model(self):
        # create fake underlying model with generate_content
        fake_model = MagicMock()
        fake_model.generate_content.return_value = "GEN-RESP"
        fake_model.model_name = "fake-model"
        wrapper = gw.GeminiWrapper(model_or_name=fake_model)
        resp = wrapper.generate_content("prompt", generation_config={"a": 1})
        self.assertEqual(resp, "GEN-RESP")
        fake_model.generate_content.assert_called_once()
        # check forwarding args passed through
        call_args, call_kwargs = fake_model.generate_content.call_args
        self.assertEqual(call_args[0], "prompt")
        self.assertIn("generation_config", call_kwargs)

    def test_start_chat_normalizes_history_and_wraps_chat_session(self):
        fake_model = MagicMock()
        # fake_model.start_chat should accept the normalized history
        fake_chat = MagicMock()
        fake_chat.history = ["a"]
        fake_model.start_chat.return_value = fake_chat
        wrapper = gw.GeminiWrapper(model_or_name=fake_model)
        # pass list-of-lists style history
        session = wrapper.start_chat(history=[["u", "m"]])
        # ensure underlying start_chat was called with normalized StrictContentType list
        called_args, called_kwargs = fake_model.start_chat.call_args
        passed_history = called_kwargs.get("history", called_args[0] if called_args else None)
        self.assertIsInstance(passed_history, list)
        self.assertEqual(len(passed_history), 2)
        self.assertTrue(all(isinstance(h, MockStrictContentType) for h in passed_history))
        # returned object should be our wrapper chat type
        self.assertIsInstance(session, gw.GeminiChatSessionWrapper)

    def test_model_not_set_raises_runtime_error(self):
        # instantiate with a string -> _model remains None
        wrapper = gw.GeminiWrapper(model_or_name="model-name-string")
        with self.assertRaises(RuntimeError):
            wrapper.generate_content("x")
        with self.assertRaises(RuntimeError):
            wrapper.start_chat()
        # model_name property should also raise
        with self.assertRaises(RuntimeError):
            _ = wrapper.model_name

    def test_model_name_returns_underlying_model_name(self):
        fake_model = MagicMock()
        fake_model.model_name = "my-model"
        wrapper = gw.GeminiWrapper(model_or_name=fake_model)
        self.assertEqual(wrapper.model_name, "my-model")


if __name__ == "__main__":
    unittest.main()
