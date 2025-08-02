import time
import os
import gradio as gr
from typing import Optional
from rag_chat import RagChat


# --- Chat Tab ---
def build_chat_tab(title: str, default_tab: str):
    with gr.Tab(label="Chat", id="Chat", interactive=(default_tab == "Chat")) as chat_tab:
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column(scale=3):
                        title_md = gr.Markdown("## " + title)
                        gr.Markdown("Chat on the left. It will cite sources from the retrieved quotes on the right.")
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

        # implement clear button
        clear.click(lambda: ([], "", ""), None, [chatbot, retrieved_quotes_box, raw_quotes_box,
                                                 research_quote_box], queue=False)

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


# --- Load Tab ---
def build_load_tab(default_tab: str):
    with gr.Tab(label="Load", id="Load", interactive=(default_tab == "Chat")) as load_tab:
        gr.Markdown("Drag and drop your files here to load them into the database.")
        gr.Markdown("Supported file types: PDF and EPUB.")
        file_input = gr.File(file_count="multiple", label="Upload a file", interactive=True)
        load_button = gr.Button("Load")
    return {
        "load_tab": load_tab,
        "file_input": file_input,
        "load_button": load_button,
    }


# --- Config Tab ---
def build_config_tab(config_data: dict):
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
                        label="Chat Title",
                        placeholder="Enter the title for the chat",
                        value=config_data["title"],
                        interactive=True,
                    )
                    sys_inst_box_tb = gr.Textbox(
                        label="System Instructions",
                        placeholder="Enter your system instructions here",
                        value=config_data["system_instructions"],
                        interactive=True,
                    )
                gr.Markdown("### API Keys")
                with gr.Group():
                    google_secret_tb = gr.Textbox(
                        label="Gemini API Key",
                        placeholder="Enter your Gemini API key here",
                        value=config_data["google_password"],
                        type="password",
                        interactive=True,
                    )
                gr.Markdown("### Postgres Settings")
                with gr.Group():
                    postgres_secret_tb = gr.Textbox(
                        label="Postgres Password",
                        placeholder="Enter your Postgres password here",
                        value=config_data["postgres_password"],
                        type="password",
                        interactive=True,
                    )
                    postgres_user_tb = gr.Textbox(
                        label="Postgres User",
                        placeholder="Enter your Postgres user here",
                        value=config_data["postgres_user_name"],
                        interactive=True,
                    )
                    postgres_db_tb = gr.Textbox(
                        label="Postgres DB",
                        placeholder="Enter your Postgres DB name here",
                        value=config_data["postgres_db_name"],
                        interactive=True,
                    )
                    postgres_table_tb = gr.Textbox(
                        label="Postgres Table",
                        placeholder="Enter your Postgres table name here",
                        value=config_data["postgres_table_name"],
                        interactive=True,
                    )
                    postgres_host_tb = gr.Textbox(
                        label="Postgres Host",
                        placeholder="Enter your Postgres host here",
                        value=config_data["postgres_host"],
                        interactive=True,
                    )
                    postgres_port_tb = gr.Textbox(
                        label="Postgres Port",
                        placeholder="Enter your Postgres port here",
                        value=str(config_data["postgres_port"]),
                        interactive=True,
                    )
    return {
        "config_tab": config_tab,
        "save_settings": save_settings,
        "chat_title_tb": chat_title_tb,
        "sys_inst_box_tb": sys_inst_box_tb,
        "google_secret_tb": google_secret_tb,
        "postgres_secret_tb": postgres_secret_tb,
        "postgres_user_tb": postgres_user_tb,
        "postgres_db_tb": postgres_db_tb,
        "postgres_table_tb": postgres_table_tb,
        "postgres_host_tb": postgres_host_tb,
        "postgres_port_tb": postgres_port_tb,
    }


def load_rag_chat(google_secret_param: str,
                  postgres_password_param: str,
                  postgres_user_name_param: str,
                  postgres_db_name_param: str,
                  postgres_table_name_param: str,
                  postgres_host_param: str,
                  postgres_port_param: int,
                  system_instructions_param: str,
                  model_name: str) -> RagChat:
    return RagChat(
        google_secret=google_secret_param,
        postgres_password=postgres_password_param,
        postgres_user_name=postgres_user_name_param,
        postgres_db_name=postgres_db_name_param,
        postgres_table_name=postgres_table_name_param,
        postgres_host=postgres_host_param,
        postgres_port=postgres_port_param,
        system_instruction=system_instructions_param,
        model_name=model_name,
    )


# noinspection PyShadowingNames
def load_config_data():
    google_password: str = ""
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
                google_password = lines[0].strip()
                postgres_password = lines[1].strip()
                postgres_user_name = lines[2].strip()
                postgres_db_name = lines[3].strip()
                postgres_table_name = lines[4].strip()
                postgres_host = lines[5].strip()
                postgres_port = int(lines[6].strip())
                title = lines[7].strip()
                system_instructions = lines[8].strip()
    return {
        "google_password": google_password,
        "postgres_password": postgres_password,
        "postgres_user_name": postgres_user_name,
        "postgres_db_name": postgres_db_name,
        "postgres_table_name": postgres_table_name,
        "postgres_host": postgres_host,
        "postgres_port": int(postgres_port),
        "system_instructions": system_instructions,
        "title": title,
    }


def build_interface(title: str = 'RAG Chat',
                    system_instructions: str = "You are a helpful assistant.",
                    model_name="gemini-2.0-flash") -> gr.Interface:

    def load_event():
        nonlocal config_data, rag_chat
        load_config()
        # Return an update for each Textbox in the same order as the outputs list below.
        return (
            gr.update(value=config_data["title"]),
            gr.update(value=config_data["system_instructions"]),
            gr.update(value=config_data["google_password"]),
            gr.update(value=config_data["postgres_password"]),
            gr.update(value=config_data["postgres_user_name"]),
            gr.update(value=config_data["postgres_db_name"]),
            gr.update(value=config_data["postgres_table_name"]),
            gr.update(value=config_data["postgres_host"]),
            gr.update(value=str(config_data["postgres_port"])),
            gr.update(interactive=(rag_chat is not None)),
            gr.update(interactive=(rag_chat is not None)),
            gr.update(selected="Chat" if rag_chat is not None else "Config"),
        )

    def load_config():
        nonlocal rag_chat, config_data, title, system_instructions
        # Load the config data from the file
        config_data = load_config_data()
        if config_data["title"] is None or config_data["title"] == "":
            config_data["title"] = title
        if config_data["system_instructions"] is None or config_data["system_instructions"] == "":
            config_data["system_instructions"] = system_instructions

        # Check if google_password and postgres_password are not empty
        if not (config_data["google_password"] == "" or config_data["postgres_password"] == "") and rag_chat is None:
            # Attempt to load RagChat with loaded values
            try:
                rag_chat = load_rag_chat(config_data["google_password"],
                                         config_data["postgres_password"],
                                         config_data["postgres_user_name"],
                                         config_data["postgres_db_name"],
                                         config_data["postgres_table_name"],
                                         config_data["postgres_host"],
                                         int(config_data["postgres_port"]),
                                         config_data["system_instructions"],
                                         model_name=model_name)
            except Exception as e:
                print(f"Error loading RagChat: {e}")
                rag_chat = None
        # If RagChat was not loaded (None) then simply return the default values

    rag_chat: Optional[RagChat] = None
    config_data: dict = {}
    load_config()
    default_tab: str = "Chat"
    if not rag_chat:
        # No config settings yet, so set Config tab as default
        default_tab: str = "Config"

    css: str = """
    #QuoteBoxes {
        height: calc(100vh - 175px);
        overflow-y: auto;
        white-space: pre-wrap;
    """
    with gr.Blocks(css=css) as chat_interface:
        with gr.Tabs(selected=default_tab) as tabs:
            chat_components = build_chat_tab(title, default_tab)
            load_components = build_load_tab(default_tab)
            config_components = build_config_tab(config_data)

        # Unpack Chat Tab components
        chat_tab = chat_components["chat_tab"]
        title_md = chat_components["title_md"]
        chatbot = chat_components["chatbot"]
        msg = chat_components["msg"]
        retrieved_quotes_box = chat_components["retrieved_quotes_box"]
        raw_quotes_box = chat_components["raw_quotes_box"]
        research_quote_box = chat_components["research_quote_box"]

        # Unpack Load Tab components
        load_tab = load_components["load_tab"]
        file_input = load_components["file_input"]
        load_button = load_components["load_button"]

        # Unpack Config Tab components
        save_settings = config_components["save_settings"]
        chat_title_tb = config_components["chat_title_tb"]
        sys_inst_box_tb = config_components["sys_inst_box_tb"]
        google_secret_tb = config_components["google_secret_tb"]
        postgres_secret_tb = config_components["postgres_secret_tb"]
        postgres_user_tb = config_components["postgres_user_tb"]
        postgres_db_tb = config_components["postgres_db_tb"]
        postgres_table_tb = config_components["postgres_table_tb"]
        postgres_host_tb = config_components["postgres_host_tb"]
        postgres_port_tb = config_components["postgres_port_tb"]

        # Attach the load event on the Blocks container:
        chat_interface.load(
            load_event,
            outputs=[
                chat_title_tb, sys_inst_box_tb, google_secret_tb, postgres_secret_tb,
                postgres_user_tb, postgres_db_tb, postgres_table_tb, postgres_host_tb,
                postgres_port_tb, chat_tab, load_tab, tabs
            ]
        )

        def user_message(message, chat_history):
            updated_history = chat_history + [(message, None)]
            return "", updated_history

        def process_message(message, chat_history):
            for updated_history, ranked_docs, all_docs, research_docs in rag_chat.respond(message, chat_history):
                yield updated_history, ranked_docs.strip(), all_docs.strip(), research_docs

        def process_with_custom_progress(files, progress=gr.Progress()):
            if files is None or len(files) == 0:
                # If no files, immediately yield cleared file list and 0% progress.
                return

            # Call the load_documents method, which now yields progress (a float between 0 and 1)
            file_enumerator = rag_chat.load_documents(files)
            for i, file in enumerate(files):
                file_name = os.path.basename(file)
                desc = f"Processing {file_name}"
                prog = i / len(files)
                progress(prog, desc=desc)
                next(file_enumerator)
            progress(1.0, desc="Finished processing")
            time.sleep(0.5)
            return "Finished processing"

        def update_progress(files):
            # Process the files and return a progress message along with an empty list to clear the widget
            process_with_custom_progress(files)
            return []

        def update_config(google_password_param: str,
                          postgres_password_param: str,
                          postgres_user_name_param: str,
                          postgres_db_name_param: str,
                          postgres_table_name_param: str,
                          postgres_host_param: str,
                          postgres_port_param: str,
                          title_param: str,
                          system_instructions_param: str):
            nonlocal rag_chat

            # Save the settings to a file
            with open("config.txt", "w") as file:
                file.write(f"{google_password_param}\n")
                file.write(f"{postgres_password_param}\n")
                file.write(f"{postgres_user_name_param}\n")
                file.write(f"{postgres_db_name_param}\n")
                file.write(f"{postgres_table_name_param}\n")
                file.write(f"{postgres_host_param}\n")
                file.write(f"{int(postgres_port_param)}\n")
                file.write(f"{title_param}\n")
                file.write(f"{system_instructions_param}\n")

            # Reset the RagChat instance with the new settings
            rag_chat = load_rag_chat(google_password_param,
                                     postgres_password_param,
                                     postgres_user_name_param,
                                     postgres_db_name_param,
                                     postgres_table_name_param,
                                     postgres_host_param,
                                     int(postgres_port_param),
                                     system_instructions_param,
                                     model_name=model_name)

            return (
                google_password_param,
                postgres_password_param,
                postgres_user_name_param,
                postgres_db_name_param,
                postgres_table_name_param,
                postgres_host_param,
                postgres_port_param,
                title_param,
                system_instructions_param,
                "## " + title_param,
                gr.update(interactive=True),
                gr.update(interactive=True),
            )

        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=True)
        msg.submit(process_message, [msg, chatbot],
                   [chatbot, retrieved_quotes_box, raw_quotes_box, research_quote_box], queue=True)

        load_button.click(update_progress, inputs=file_input, outputs=file_input)

        save_settings.click(update_config,
                            inputs=[google_secret_tb, postgres_secret_tb, postgres_user_tb, postgres_db_tb,
                                    postgres_table_tb, postgres_host_tb, postgres_port_tb, chat_title_tb,
                                    sys_inst_box_tb],
                            outputs=[
                                google_secret_tb, postgres_secret_tb, postgres_user_tb, postgres_db_tb,
                                postgres_table_tb, postgres_host_tb, postgres_port_tb, chat_title_tb,
                                sys_inst_box_tb, title_md, chat_tab, load_tab,
                            ],
                            queue=True)

    return chat_interface


if __name__ == "__main__":
    sys_instruction: str = ("You are philosopher Karl Popper. Answer questions with philosophical insights, and use "
                            "the provided quotes along with their metadata as reference.")
    # gemma-3-27b-it, gemini-2.0-flash, gemini-2.0-flash-exp, gemini-1.5-flash
    rag_chat_ui = build_interface(title="Karl Popper Chatbot",
                                  system_instructions=sys_instruction,
                                  model_name="gemini-2.0-flash")
    rag_chat_ui.launch(debug=True, max_file_size=100 * gr.FileSize.MB)
