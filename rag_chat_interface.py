# *** top of file, before importing torch or transformers ***
import os
import time
import gradio as gr
from typing import Optional, Union
from rag_chat import RagChat
from models.llm_model import LLMModel
from utilities.general_utils import get_secret


class RAGChatInterface:
    def __init__(
        self,
        model_or_name: Union[LLMModel, str],
        default_title: str = "RAG Chat",
        embeder_model_name: str = "BAAI/llm-embedder",
        postgres_table_recreate: bool = False,
        llm_top_k: int = 5,
        retriever_top_k_docs=100
    ):
        self._title: str = default_title
        self._model_or_model_name: Union[LLMModel, str] = model_or_name
        self._rag_chat: Optional[RagChat] = None
        self._config_data: dict = {}
        self._llm_top_k: int = llm_top_k
        self._retriever_top_k_docs: int = retriever_top_k_docs
        self._embeder_model_name: str = embeder_model_name
        self._postgres_table_recreate: bool = postgres_table_recreate

    # TODO: Fix streaming so that it works even without streaming (stream=False)
    def init_or_update_rag_chat(self,
                                model_password: Optional[str],
                                postgres_password: str,
                                *,
                                stream: bool = True,
                                postgres_user_name: str,
                                postgres_db_name: str,
                                postgres_table_name: str,
                                postgres_host: str,
                                postgres_port: int,
                                postgres_table_recreate: bool,
                                embedder_model_name: str,
                                system_instructions: Optional[str],
                                llm_top_k: int,
                                retriever_top_k_docs: int) -> None:

        try:
            model: LLMModel
            if isinstance(self._model_or_model_name, str):
                model = LLMModel(model_or_name=self._model_or_model_name,
                                 system_instruction=system_instructions,
                                 secret_token=model_password)
            else:
                model = self._model_or_model_name

            if self._rag_chat is None:
                self._rag_chat = RagChat(
                    model,
                    stream=stream,
                    postgres_password=postgres_password,
                    postgres_user_name=postgres_user_name,
                    postgres_db_name=postgres_db_name,
                    postgres_table_name=postgres_table_name,
                    postgres_host=postgres_host,
                    postgres_port=int(postgres_port),
                    llm_top_k=llm_top_k,
                    retriever_top_k_docs=retriever_top_k_docs,
                    embedder_model_name=embedder_model_name,
                    postgres_table_recreate=postgres_table_recreate,
                )
            else:
                self._rag_chat = self._rag_chat.update_rag_chat(
                    model=model,
                    stream=stream,
                    postgres_password=postgres_password,
                    postgres_user_name=postgres_user_name,
                    postgres_db_name=postgres_db_name,
                    postgres_table_name=postgres_table_name,
                    postgres_host=postgres_host,
                    postgres_port=int(postgres_port),
                    llm_top_k=llm_top_k,
                    retriever_top_k_docs=retriever_top_k_docs,
                    embedder_model_name=embedder_model_name,
                    postgres_table_recreate=postgres_table_recreate,
                )

        except Exception as e:
            print(f"Error loading RagChat: {e}")
            self._rag_chat = None

    @staticmethod
    def build_chat_tab(title: str, default_tab: str) -> dict:
        with gr.Tab(label="Chat", id="Chat", interactive=(default_tab == "Chat")) as chat_tab:
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column(scale=3):
                            title_md = gr.Markdown("## " + title)
                            gr.Markdown("Chat on the left. "
                                        "It will cite sources from the retrieved quotes on the right.")
                        with gr.Column(scale=1):
                            clear = gr.Button("Clear Chat")
                    chatbot = gr.Chatbot(label="Chat")
                    msg = gr.Textbox(placeholder="Ask your question...", label="Your Message")
                with gr.Column(scale=1):
                    with gr.Tab("Retrieved Quotes"):
                        retrieved_quotes_box = gr.Markdown(
                            label="Retrieved Quotes & Metadata", value="", elem_id="QuoteBoxes"
                        )
                    with gr.Tab("Raw Quotes"):
                        raw_quotes_box = gr.Markdown(
                            label="Raw Quotes & Metadata", value="", elem_id="QuoteBoxes"
                        )
                    with gr.Tab("Research"):
                        research_quote_box = gr.Markdown(
                            label="Quotes found by LLM during research", value="", elem_id="QuoteBoxes"
                        )

            clear.click(
                lambda: ([], "", "", ""), None,
                [chatbot, retrieved_quotes_box, raw_quotes_box, research_quote_box],
                queue=False
            )

        return {
            "chat_tab": chat_tab,
            "title_md": title_md,
            "clear": clear,
            "chatbot": chatbot,
            "msg": msg,
            "retrieved_quotes_box": retrieved_quotes_box,
            "raw_quotes_box": raw_quotes_box,
            "research_quote_box": research_quote_box,
        }

    @staticmethod
    def build_load_tab(default_tab: str) -> dict:
        with gr.Tab(label="Load", id="Load", interactive=(default_tab == "Chat")) as load_tab:
            gr.Markdown("Drag and drop your files here to load them into the database.")
            gr.Markdown("Supported file types: PDF and EPUB.")
            file_input = gr.File(file_count="multiple", label="Upload a file", interactive=True)
            load_button = gr.Button("Load")

        return {"load_tab": load_tab, "file_input": file_input, "load_button": load_button}

    @staticmethod
    def build_config_tab(config_data: dict) -> dict:
        with gr.Tab(label="Config", id="Config") as config_tab:
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("Settings for chat and load.")
                    gr.Markdown("### Chat Settings")
                with gr.Column(scale=1):
                    save_settings = gr.Button("Save Settings")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        chat_title_tb = gr.Textbox(
                            label="Chat Title", placeholder="Enter the title for the chat",
                            value=config_data["title"], interactive=True,
                        )
                        sys_inst_box_tb = gr.Textbox(
                            label="System Instructions", placeholder="Enter your system instructions here",
                            value=config_data["system_instructions"], interactive=True,
                        )
                    gr.Markdown("### Model Secret")
                    with gr.Group():
                        model_secret_cb = gr.Checkbox(
                            label="Model needs password", value=True, interactive=True
                        )
                        model_secret_tb = gr.Textbox(
                            label="Model Secret", placeholder="Enter your model secret "
                                                              "(e.g. Gemini API key or Hugging Face key) here",
                            value=config_data["model_password"], type="password", interactive=True,
                        )
                    gr.Markdown("### Postgres Settings")
                    with gr.Group():
                        postgres_secret_tb = gr.Textbox(
                            label="Postgres Password", placeholder="Enter your Postgres password here",
                            value=config_data["postgres_password"], type="password", interactive=True,
                        )
                        postgres_user_tb = gr.Textbox(
                            label="Postgres User", placeholder="Enter your Postgres user here",
                            value=config_data["postgres_user_name"], interactive=True,
                        )
                        postgres_db_tb = gr.Textbox(
                            label="Postgres DB", placeholder="Enter your Postgres DB name here",
                            value=config_data["postgres_db_name"], interactive=True,
                        )
                        postgres_table_tb = gr.Textbox(
                            label="Postgres Table", placeholder="Enter your Postgres table name here",
                            value=config_data["postgres_table_name"], interactive=True,
                        )
                        postgres_host_tb = gr.Textbox(
                            label="Postgres Host", placeholder="Enter your Postgres host here",
                            value=config_data["postgres_host"], interactive=True,
                        )
                        postgres_port_tb = gr.Textbox(
                            label="Postgres Port", placeholder="Enter your Postgres port here",
                            value=str(config_data["postgres_port"]), interactive=True,
                        )

        return {
            "config_tab": config_tab,
            "save_settings": save_settings,
            "chat_title_tb": chat_title_tb,
            "sys_inst_box_tb": sys_inst_box_tb,
            "model_secret_cb": model_secret_cb,
            "model_secret_tb": model_secret_tb,
            "postgres_secret_tb": postgres_secret_tb,
            "postgres_user_tb": postgres_user_tb,
            "postgres_db_tb": postgres_db_tb,
            "postgres_table_tb": postgres_table_tb,
            "postgres_host_tb": postgres_host_tb,
            "postgres_port_tb": postgres_port_tb,
        }

    def init_chat_config_tabs(self) -> tuple[Optional[RagChat], dict[str, str]]:
        config_data = load_config_data()
        if not config_data["title"]:
            config_data["title"] = self._title

        if config_data["model_password"] and config_data["postgres_password"] and self._rag_chat is None:
            self.init_or_update_rag_chat(
                model_password=config_data["model_password"] if config_data["needs_model_password"] else None,
                postgres_password=config_data["postgres_password"],
                postgres_user_name=config_data["postgres_user_name"],
                postgres_db_name=config_data["postgres_db_name"],
                postgres_table_name=config_data["postgres_table_name"],
                postgres_host=config_data["postgres_host"],
                postgres_port=int(config_data["postgres_port"]),
                postgres_table_recreate=False,
                embedder_model_name=self._embeder_model_name,
                llm_top_k=self._llm_top_k,
                retriever_top_k_docs=self._retriever_top_k_docs,
                system_instructions=config_data["system_instructions"],
            )

        return self._rag_chat, config_data

    # Refactored sub-functions to class methods
    def load_event(self) -> tuple:
        self._rag_chat, self._config_data = self.init_chat_config_tabs()
        return (
            gr.update(value=self._config_data["title"]),
            gr.update(value=self._config_data["system_instructions"]),
            gr.update(value=self._config_data["needs_model_password"] != ""),
            gr.update(value=self._config_data["model_password"]),
            gr.update(value=self._config_data["postgres_password"]),
            gr.update(value=self._config_data["postgres_user_name"]),
            gr.update(value=self._config_data["postgres_db_name"]),
            gr.update(value=self._config_data["postgres_table_name"]),
            gr.update(value=self._config_data["postgres_host"]),
            gr.update(value=str(self._config_data["postgres_port"])),
            gr.update(interactive=(self._rag_chat is not None)),
            gr.update(interactive=(self._rag_chat is not None)),
            gr.update(selected="Chat" if self._rag_chat else "Config"),
        )

    @staticmethod
    def handle_user_message(message, chat_history):
        return "", chat_history + [(message, None)]

    def process_message(self, message, chat_history):
        for updated_history, ranked_docs, all_docs, research_docs in self._rag_chat.respond(message, chat_history):
            yield updated_history, ranked_docs.strip(), all_docs.strip(), research_docs

    def process_with_custom_progress(self, files, progress=gr.Progress()):
        if not files:
            return
        file_enumerator = self._rag_chat.load_documents(files)
        for i, file in enumerate(files):
            progress(i / len(files), desc=f"Processing {os.path.basename(file)}")
            next(file_enumerator)
        progress(1.0, desc="Finished processing")
        time.sleep(0.5)
        return "Finished processing"

    def update_progress(self, files):
        self.process_with_custom_progress(files)
        return []

    def update_config(self,
                      needs_model_password_param,
                      model_password_param,
                      postgres_password_param,
                      postgres_user_name_param,
                      postgres_db_name_param,
                      postgres_table_name_param,
                      postgres_host_param,
                      postgres_port_param,
                      title_param,
                      system_instructions_param):
        with open("config.txt", "w") as file:
            file.write(f"{bool(needs_model_password_param)}\n")
            file.write(f"{model_password_param}\n")
            file.write(f"{postgres_password_param}\n")
            file.write(f"{postgres_user_name_param}\n")
            file.write(f"{postgres_db_name_param}\n")
            file.write(f"{postgres_table_name_param}\n")
            file.write(f"{postgres_host_param}\n")
            file.write(f"{int(postgres_port_param)}\n")
            file.write(f"{title_param}\n")
            file.write(f"{system_instructions_param}\n")

        self.init_or_update_rag_chat(
            model_password=model_password_param if needs_model_password_param else None,
            postgres_password=postgres_password_param,
            postgres_user_name=postgres_user_name_param,
            postgres_db_name=postgres_db_name_param,
            postgres_table_name=postgres_table_name_param,
            postgres_host=postgres_host_param,
            postgres_port=int(postgres_port_param),
            postgres_table_recreate=self._postgres_table_recreate,
            embedder_model_name=self._embeder_model_name,
            system_instructions=system_instructions_param,
            llm_top_k=self._llm_top_k,
            retriever_top_k_docs=self._retriever_top_k_docs,
        )

        return (
            model_password_param, postgres_password_param,
            postgres_user_name_param, postgres_db_name_param,
            postgres_table_name_param, postgres_host_param,
            postgres_port_param, title_param,
            system_instructions_param, "## " + title_param,
            gr.update(interactive=True), gr.update(interactive=True),
        )

    def build_interface(self) -> gr.Interface:
        # Initialize chat config and determine default tab
        self._rag_chat, self._config_data = self.init_chat_config_tabs()
        default_tab = "Chat" if self._rag_chat else "Config"

        css = """
        #QuoteBoxes {
            height: calc(100vh - 185px);
            overflow-y: auto;
            white-space: pre-wrap;
        }
        """

        with gr.Blocks(css=css) as chat_interface:
            with gr.Tabs(selected=default_tab) as tabs:
                chat_components = self.build_chat_tab(self._title, default_tab)
                load_components = self.build_load_tab(default_tab)
                config_components = self.build_config_tab(self._config_data)

            # Unpack components
            chat_tab = chat_components["chat_tab"]
            title_md = chat_components["title_md"]
            chatbot = chat_components["chatbot"]
            msg = chat_components["msg"]
            retrieved_quotes_box = chat_components["retrieved_quotes_box"]
            raw_quotes_box = chat_components["raw_quotes_box"]
            research_quote_box = chat_components["research_quote_box"]

            load_tab = load_components["load_tab"]
            file_input = load_components["file_input"]
            load_button = load_components["load_button"]

            save_settings = config_components["save_settings"]
            chat_title_tb = config_components["chat_title_tb"]
            sys_inst_box_tb = config_components["sys_inst_box_tb"]
            model_secret_cb = config_components["model_secret_cb"]
            model_secret_tb = config_components["model_secret_tb"]
            postgres_secret_tb = config_components["postgres_secret_tb"]
            postgres_user_tb = config_components["postgres_user_tb"]
            postgres_db_tb = config_components["postgres_db_tb"]
            postgres_table_tb = config_components["postgres_table_tb"]
            postgres_host_tb = config_components["postgres_host_tb"]
            postgres_port_tb = config_components["postgres_port_tb"]

            # Bind events to class methods
            chat_interface.load(
                self.load_event,
                outputs=[
                    chat_title_tb, sys_inst_box_tb, model_secret_cb, model_secret_tb, postgres_secret_tb,
                    postgres_user_tb, postgres_db_tb, postgres_table_tb, postgres_host_tb,
                    postgres_port_tb, chat_tab, load_tab, tabs
                ]
            )

            msg.submit(self.handle_user_message, [msg, chatbot], [msg, chatbot], queue=True)
            msg.submit(
                self.process_message,
                [msg, chatbot],
                [chatbot, retrieved_quotes_box, raw_quotes_box, research_quote_box],
                queue=True
            )
            load_button.click(self.update_progress, inputs=file_input, outputs=file_input)
            save_settings.click(
                self.update_config,
                inputs=[
                    model_secret_cb, model_secret_tb, postgres_secret_tb, postgres_user_tb, postgres_db_tb,
                    postgres_table_tb, postgres_host_tb, postgres_port_tb, chat_title_tb,
                    sys_inst_box_tb
                ],
                outputs=[
                    model_secret_tb, postgres_secret_tb, postgres_user_tb, postgres_db_tb,
                    postgres_table_tb, postgres_host_tb, postgres_port_tb, chat_title_tb,
                    sys_inst_box_tb, title_md, chat_tab, load_tab,
                ],
                queue=True
            )

        return chat_interface


def load_config_data() -> dict[str, str]:
    needs_model_password: bool = True
    model_password: str = ""
    postgres_password: str = ""
    postgres_user_name: str = "postgres"
    postgres_db_name: str = "postgres"
    postgres_table_name: str = "book_archive"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    title: str = ""
    system_instructions: str = ""

    if os.path.exists("config.txt"):
        with open("config.txt", "r") as f:
            lines = f.readlines()
            if len(lines) >= 9:
                needs_model_password = bool(lines[0].strip())
                model_password = lines[1].strip()
                postgres_password = lines[2].strip()
                postgres_user_name = lines[3].strip()
                postgres_db_name = lines[4].strip()
                postgres_table_name = lines[5].strip()
                postgres_host = lines[6].strip()
                postgres_port = int(lines[7].strip())
                title = lines[8].strip()
                system_instructions = lines[9].strip()

    return {
        "needs_model_password": needs_model_password,
        "model_password": model_password,
        "postgres_password": postgres_password,
        "postgres_user_name": postgres_user_name,
        "postgres_db_name": postgres_db_name,
        "postgres_table_name": postgres_table_name,
        "postgres_host": postgres_host,
        "postgres_port": int(postgres_port),
        "system_instructions": system_instructions,
        "title": title,
    }


if __name__ == "__main__":
    sys_instruction = (
        "You are philosopher Karl Popper. Answer questions with philosophical insights, and use "
        "the provided quotes along with their metadata as reference."
    )
    # llm_client = LLMModel(model_or_name="gemini-2.0-flash", "google/gemma-2-2b-it", "google/gemma-3-270m"
    # system_instructions=sys_instruction)
    google_secret: str = get_secret(r'D:\Documents\Secrets\gemini_secret.txt')  # Put your path here # noqa: F841
    # llm_client = LLMModel("google/gemma-3-270m", system_instructions="You are concise.", secret_token=None)
    app = RAGChatInterface(
        model_or_name="gemini-2.0-flash",
        default_title="AI Philosopher",
        llm_top_k=3,
        retriever_top_k_docs=10,
    )
    interface = app.build_interface()
    interface.launch(debug=True, max_file_size=100 * gr.FileSize.MB)
