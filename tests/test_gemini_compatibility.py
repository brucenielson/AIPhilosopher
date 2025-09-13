import unittest
from models.gemini_compatibility import (
    normalize_safety_to_dict,
    normalize_config_to_dict,
    normalize_tool_config_to_dict,
    normalize_instruction_to_str,
    GeminiChatSessionCompatible,
    GeminiCompatible,
    ContentLike,
)


# --- Tests for Normalizers ---
class TestNormalizers(unittest.TestCase):
    def test_normalize_safety_to_dict_with_dict(self):
        input_data = {"key": "value"}
        self.assertEqual(normalize_safety_to_dict(input_data), input_data)

    def test_normalize_safety_to_dict_with_none(self):
        self.assertEqual(normalize_safety_to_dict(None), {})

    def test_normalize_config_to_dict_with_dict(self):
        input_data = {"gen": "config"}
        self.assertEqual(normalize_config_to_dict(input_data), input_data)

    def test_normalize_config_to_dict_with_none(self):
        self.assertEqual(normalize_config_to_dict(None), {})

    def test_normalize_tool_config_to_dict_with_dict(self):
        input_data = {"tool": "cfg"}
        self.assertEqual(normalize_tool_config_to_dict(input_data), input_data)

    def test_normalize_tool_config_to_dict_with_none(self):
        self.assertEqual(normalize_tool_config_to_dict(None), {})

    def test_normalize_instruction_to_str_with_str(self):
        s = "instruction"
        self.assertEqual(normalize_instruction_to_str(s), s)

    def test_normalize_instruction_to_str_with_none(self):
        self.assertEqual(normalize_instruction_to_str(None), "")


# --- Mock classes to test abstract classes ---
class MockChatSession(GeminiChatSessionCompatible[str]):
    def send_message(self, contents: ContentLike, **kwargs) -> str:
        return f"echo: {contents}"


class MockGeminiModel(GeminiCompatible[str]):
    def generate_content(self, contents: ContentLike, **kwargs) -> str:
        return f"generated: {contents}"

    def start_chat(self, history) -> GeminiChatSessionCompatible[str]:
        return MockChatSession(history)


# --- Tests for GeminiChatSessionCompatible ---
class TestGeminiChatSessionCompatible(unittest.TestCase):
    def test_chat_session_send_message_and_history(self):
        session = MockChatSession(history=["hi"])
        self.assertEqual(session.send_message("hello"), "echo: hello")
        self.assertEqual(session.get_history(), ["hi"])


# --- Tests for GeminiCompatible ---
class TestGeminiCompatible(unittest.TestCase):
    def test_gemini_compatible_properties_normalize(self):
        model = MockGeminiModel(model_or_name="test-model", normalize=True)
        self.assertEqual(model.model_name, "test-model")
        self.assertEqual(model.system_instruction, '')
        self.assertEqual(model.generation_config, {})
        self.assertEqual(model.safety_settings, {})
        self.assertEqual(model.tool_config, {})
        self.assertIsNone(model.tools)

    def test_gemini_compatible_properties_no_normalize(self):
        model = MockGeminiModel(model_or_name="test-model", normalize=False)
        self.assertEqual(model.model_name, "test-model")
        self.assertIsNone(model.system_instruction)
        self.assertIsNone(model.generation_config)
        self.assertIsNone(model.safety_settings)
        self.assertIsNone(model.tool_config)
        self.assertIsNone(model.tools)

    def test_gemini_compatible_generate_content(self):
        model = MockGeminiModel(model_or_name="test-model")
        result = model.generate_content("prompt")
        self.assertEqual(result, "generated: prompt")

    def test_gemini_compatible_start_chat_returns_session(self):
        model = MockGeminiModel(model_or_name="test-model")
        session = model.start_chat(history=["hello"])
        self.assertIsInstance(session, GeminiChatSessionCompatible)
        self.assertEqual(session.send_message("test"), "echo: test")

    def test_gemini_compatible_forwarding(self):
        # _model has no attributes, so accessing a non-existent one raises AttributeError
        model = MockGeminiModel(model_or_name="test-model")
        with self.assertRaises(AttributeError):
            _ = model.non_existent_attr


if __name__ == "__main__":
    unittest.main()
