import time
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerateContentResponse
# noinspection PyPackageRequirements
from google.generativeai import ChatSession
# noinspection PyPackageRequirements
from google.generativeai.types import generation_types
# noinspection PyPackageRequirements
import google.generativeai as genai
from doc_retrieval_pipeline import DocRetrievalPipeline, SearchMode
from document_processor import DocumentProcessor
# noinspection PyPackageRequirements
from haystack import Document
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
from react_agent import format_document, ReActAgent
from llm_message_utils import send_message


class RagChat:
    def __init__(self,
                 google_secret: str,
                 postgres_password: str,
                 postgres_user_name: str = "postgres",
                 postgres_db_name: str = "postgres",
                 postgres_table_name: str = "popper_archive",
                 postgres_host: str = 'localhost',
                 postgres_port: int = 5432,
                 postgres_table_recreate: bool = False,
                 postgres_table_embedder_model_name: str = "BAAI/llm-embedder",
                 model_name: str = "gemini-2.0-flash",
                 system_instruction: Optional[str] = None, ):

        # Initialize Gemini Chat with a system instruction to act like philosopher Karl Popper.
        self._model: Optional[genai.GenerativeModel] = None
        self._system_instruction: Optional[str] = system_instruction
        self._google_secret: str = google_secret
        self.initialize_model(model_name=model_name,
                              system_instruction=system_instruction,
                              google_secret=google_secret)

        # Initialize the document retrieval pipeline with top-5 quote retrieval.
        self._postgres_password: str = postgres_password
        self._postgres_user_name: str = postgres_user_name
        self._postgres_db_name: str = postgres_db_name
        self._postgres_table_name: str = postgres_table_name
        self._postgres_table_recreate: bool = postgres_table_recreate
        self._postgres_table_embedder_model_name: str = postgres_table_embedder_model_name
        self._postgres_host: str = postgres_host
        self._postgres_port: int = postgres_port
        # Initialize the document retrieval pipeline.
        self._doc_pipeline: DocRetrievalPipeline = DocRetrievalPipeline(
            table_name=self._postgres_table_name,
            db_user_name=self._postgres_user_name,
            db_password=self._postgres_password,
            postgres_host=self._postgres_host,
            postgres_port=self._postgres_port,
            db_name=self._postgres_db_name,
            verbose=False,
            llm_top_k=5,
            retriever_top_k_docs=100,
            include_outputs_from=None,
            search_mode=SearchMode.HYBRID,
            use_reranker=True,
            embedder_model_name=self._postgres_table_embedder_model_name,
        )
        self._load_pipeline: Optional[DocumentProcessor] = None

    def initialize_model(self, model_name: str = "gemini-2.0-flash",
                         system_instruction: Optional[str] = None,
                         google_secret: Optional[str] = None):
        genai.configure(api_key=google_secret)
        self._google_secret = google_secret

        if 'gemma' in model_name:
            # If using Gemma, set the system instruction to None as it does not support it.
            system_instruction = None

        model: genai.GenerativeModel = genai.GenerativeModel(
            model_name=model_name,  # gemini-2.0-flash-exp, gemini-2.0-flash, gemma-3-27b-it
            system_instruction=system_instruction
        )
        self._model = model

    def ask_llm_question(self, prompt: str,
                         chat_history: Optional[List[Dict[str, Any]]] = None,
                         stream: bool = False) -> Union[generation_types.GenerateContentResponse, str]:
        if chat_history is None:
            chat_history = []
        if self._google_secret is not None and self._google_secret != "":
            # Start a new chat session with no history for this check.
            # chat_session = self._model.start_chat(history=chat_history)
            # chat_response = chat_session.send_message(prompt, stream=stream)
            chat_session: ChatSession = self._model.start_chat(history=chat_history)
            chat_response: GenerateContentResponse = send_message(chat_session, prompt, stream=stream)
            # If streaming is enabled, return the response object.
            if stream:
                return chat_response
            # If streaming is not enabled, return the full response text.
            else:
                return chat_response.text.strip()
        else:
            # If no secret is provided, throw an error
            raise ValueError("Google secret is not provided. Please provide a valid API key.")

    def ask_llm_for_quote_relevance(self, message: str, docs: List[Document]) -> str:
        """
        Given a question and a list of retrieved documents, ask the LLM to determine which quotes are relevant.
        This function formats the question and documents into a prompt for the LLM, which will then return a list of
        relevant quote numbers based on order of Documents in the docs list. e.g. "1,3,5".
        You can then use this list to filter the documents to which the LLM found most relevant to the question.

        Args:
            message (str): The question or prompt to evaluate.
            docs (List[Document]): The list of documents containing quotes to be reviewed.

        Returns:
            str: A comma-separated list of numbers indicating the relevant quotes, or an empty string if none are
            relevant.
        """
        prompt = (
            f"Given the question: '{message}', review the following numbered quotes and "
            "return a comma-separated list of the numbers for the quotes that you believe will help answer the "
            "question. If there are no quotes relevant to the question, return an empty string. "
            "Answer with only the numbers or an empty string, for example: '1,3,5' or ''.\n\n"
        )
        for i, doc in enumerate(docs, start=1):
            prompt += f"{i}. {doc.content}\n\n"

        return self.ask_llm_question(prompt)

    def ask_llm_to_research(self, users_question: str) -> Tuple[str, List[Document]]:
        # Instantiate the ReActFunctionCaller session using the defined model.
        gemini_react: ReActAgent = ReActAgent(doc_retriever=self._doc_pipeline)
        return gemini_react(users_question, temperature=0.2)

    def ask_llm_for_improved_query(self, message: str, chat_history: List[Dict[str, Any]]) -> str:
        prompt = (
            f"Given the query: '{message}' and the current chat history, the database of relevant quotes found none "
            f"that were a strong match. This might be due to poor wording on the user's part. "
            f"Reviewing the query and chat history, determine if you can provide a better wording for the query "
            f"that might yield better results. If you can improve the query, return the improved "
            f"query. If you cannot improve the question, return an empty string (without quotes around it) and we'll "
            f"continue with the user's original query. There is no need to explain your thinking if you want to return "
            f"an empty string. Do not return quotes around your answer.\n\n"
            f"You must either return a single sentence or phrase that is the new query (without quotes around it) or "
            f"an empty string (without quotes around it). Keep the new query as concise as possible to improve matches."
            f"\n\n"
        )
        improved_query: str = self.ask_llm_question(prompt, chat_history)
        # The LLM is sometimes stupid and takes my example too literally and returns "''" instead of "" for an
        # empty string. So we need to check for that and convert it to an empty string.
        # Unfortunately, dropping that instruction tends to cause it to think out loud before returning an empty
        # string at the end. Which sort of defeats the purpose.

        # Strip off double or single quotes if the improved query starts and ends with them.
        if improved_query.startswith(('"', "'")) and improved_query.endswith(('"', "'")):
            improved_query = improved_query[1:-1]
        if improved_query.lower() == "empty string":
            improved_query = ""

        return improved_query

    @staticmethod
    def get_max_score(docs: Optional[List[Document]]) -> float:
        """
        Get the maximum score from a list of documents.

        Args:
            docs (List[Document]): The list of documents to evaluate.

        Returns:
            float: The maximum score found in the documents.
        """
        # Find the largest score
        max_score: float = 0.0
        if docs:
            max_score = max(doc.score for doc in docs if hasattr(doc, 'score'))
        return max_score

    def load_documents(self, files: List[str]) -> Iterator[None]:
        if self._load_pipeline is None:
            self._load_pipeline: DocumentProcessor = DocumentProcessor(
                table_name=self._postgres_table_name,
                recreate_table=False,
                embedder_model_name="BAAI/llm-embedder",
                file_folder_path_or_list=files,
                db_user_name=self._postgres_user_name,
                db_password=self._postgres_password,
                postgres_host='localhost',
                postgres_port=5432,
                db_name=self._postgres_db_name,
                min_section_size=3000,
                min_paragraph_size=300,
            )
        # Load the documents into the database.
        for _ in self._load_pipeline.run(files, use_iterator=True):
            yield

    def respond(self, message: Optional[str], chat_history: List[Optional[List[str]]]):
        # --- Step 1: Retrieve the top-5 quotes with metadata ---
        if message.strip() == "" or message is None:
            # This is a kludge to deal with the fact that Gradio sometimes get a race condition, and we lose the message
            # To correct, try to get the last message from chat history
            if chat_history and len(chat_history) > 0 and chat_history[-1][1] is None:
                # If the last message has no response, then grab the message portion and remove it.
                # It will get added back again below.
                # There has got to be a better way to do this, but this will work for now
                message = chat_history[-1][0]
                # Remove last message from chat history
                chat_history = chat_history[:-1]

        # Put the chat_history into the correct format for Gemini
        gemini_chat_history: List[Dict[str, Any]] = self.transform_history(chat_history)

        retrieved_docs: List[Document]
        all_docs: List[Document]
        research_response: Optional[str] = None
        research_docs: List[Document] = []
        retrieved_docs, all_docs = self._doc_pipeline.generate_response(message)

        # Find the largest score
        max_score: float = self.get_max_score(retrieved_docs)

        if max_score is not None and max_score < 0.50:
            # If we don't have any good quotes, ask the LLM if it wants to do its own search
            improved_query: str = self.ask_llm_for_improved_query(message, gemini_chat_history)

            new_retrieved_docs: List[Document]
            temp_all_docs: List[Document]
            if improved_query != "":
                new_retrieved_docs, temp_all_docs = self._doc_pipeline.generate_response(improved_query)
                new_max_score: float = self.get_max_score(new_retrieved_docs)
                if new_max_score > max(max_score * 1.1, max_score + 0.05):
                    # If the new max score is better than the old one, use the new docs
                    retrieved_docs = new_retrieved_docs
                    all_docs = temp_all_docs
                    max_score = new_max_score

        if max_score is not None and max_score < 0.30:
            # If there are no quotes with a score at least 0.30,
            # then we ask Gemini in one go which quotes are relevant.
            response_text = self.ask_llm_for_quote_relevance(message, retrieved_docs)
            # Split by commas, remove any extra spaces, and convert to integers.
            try:
                relevant_numbers = [int(num.strip()) for num in response_text.split(',') if num.strip().isdigit()]
            except Exception as parse_e:
                print(f"Error parsing Gemini response: {parse_e}")
                time.sleep(1)
                relevant_numbers = []

            # Filter docs based on the numbered positions.
            ranked_docs = [doc for idx, doc in enumerate(retrieved_docs, start=1) if idx in relevant_numbers]
        else:
            # Drop any quotes with a score less than 0.20 if we have at least 3 quotes above 20
            # Otherwise drop any quotes with a score less than 0.10
            # Count how many quotes have a score >= 0.20.
            threshold: float = 0.20
            num_high = len([doc for doc in retrieved_docs if hasattr(doc, 'score') and doc.score >= threshold])
            # If we have at least 3 such quotes, drop any with a score less than 0.20.
            # Otherwise, drop quotes with a score less than 0.10.
            threshold = 0.20 if num_high >= 3 else 0.10
            ranked_docs = [doc for doc in retrieved_docs if hasattr(doc, 'score') and doc.score >= threshold]

        # If we have under 5 quotes or the max score is under 0.6 then attempt ReAct research
        if len(ranked_docs) < 5 or max_score < 0.6:
            # Ask the LLM to do its own research
            research_response, research_docs = self.ask_llm_to_research(message)

        # Format each retrieved document (quote + metadata).
        formatted_docs = [format_document(doc) for doc in ranked_docs]
        retrieved_quotes = "\n\n".join(formatted_docs)
        formatted_docs = [format_document(doc, include_raw_info=True) for doc in all_docs]
        all_quotes = "\n\n".join(formatted_docs)
        research_quotes = "\n\n".join([format_document(doc) for doc in research_docs])

        modified_query: str
        if not retrieved_quotes or retrieved_quotes.strip() == "":
            # We have no quotes to use, so ask for an answer without quotes.
            modified_query = (
                f"Answer the following question: {message}\n"
                f"Use the following research response as reference:\n{research_response}\n\n"
            )
        elif max_score > 0.50:
            modified_query = (
                f"Use the following quotes with their metadata as reference in your answer:\n\n{retrieved_quotes}\n\n"
                f"Reference the quotes and their metadata in your answer where possible. "
                f"Now, answer the following question: {message}"
            )
        else:
            modified_query = (
                f"The following quotes are available. You may use them as reference to answer my question"
                f"if you find them relevant:\n\n{retrieved_quotes}\n\n"
                f"Reference the quotes and their metadata in your answer if used. But don't "
                f"feel obligated to use the quotes if they are not relevant. "
                f"Now, answer the following question: {message}"
            )

        if research_response is not None and research_response.strip() != "":
            # We have a research response so tack that on to the end of the query
            modified_query += (f"\nYou may use your researched response to help answer the question:\n"
                               f"{research_response}\n\n")

        # We start a new chat session each time so that we can control the chat history and remove all the rag docs
        # Send the modified query to Gemini.
        chat_response = self.ask_llm_question(modified_query, chat_history=gemini_chat_history, stream=True)
        answer_text = ""
        # # --- Step 3: Stream the answer character-by-character ---
        for chunk in chat_response:
            try:
                if hasattr(chunk, 'text'):
                    answer_text += chunk.text
                    yield chat_history + [(message, answer_text)], retrieved_quotes, all_quotes, research_quotes
            except ValueError:
                # Gemma seems to have some bad responses that cause a ValueError when trying to access
                # So skip over those.
                continue

    # Taken from https://medium.com/latinxinai/simple-chatbot-gradio-google-gemini-api-4ce02fbaf09f
    @staticmethod
    def transform_history(history) -> List[Dict[str, Any]]:
        new_history = []
        for chat_response in history:
            new_history.append({"parts": [{"text": chat_response[0]}], "role": "user"})
            new_history.append({"parts": [{"text": chat_response[1]}], "role": "model"})
        return new_history
