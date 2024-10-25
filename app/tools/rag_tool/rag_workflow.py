import os
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# resp = Gemini(model="models/gemini-1.5-flash-latest").complete("Write a poem about a magic backpack")
# print(resp)

# model_name = "models/embedding-001"

# embed_model = GeminiEmbedding(
#     model_name=model_name, api_key=GOOGLE_API_KEY, title="this is a document"
# )

# embeddings = embed_model.get_text_embedding("Google Gemini Embeddings.")

# print(embeddings)

class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]
    

class RAGWorkflow(Workflow):
    
    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        index = ev.get("index")

        if not query:
            return None

        print(f"Query the database with: {query}")

        # get the index from the global context
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        nodes = index.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")

        await ctx.set("query", query)
        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""

        template = (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information, please answer the question: {query_str}\n"
        )

        qa_template = PromptTemplate(template)

        # llm = Ollama(model="gemma2:2b", request_timeout=60.0) # phi3:mini
        llm = Gemini(model="models/gemini-1.5-flash-latest")
        summarizer = CompactAndRefine(llm=llm, text_qa_template=qa_template)

        query = await ctx.get("query", default=None)

        response = summarizer.synthesize(query, nodes=ev.nodes)

        return StopEvent(result=response)