from transformers.agents.tools import Tool
from data import Data


class SquadRetrieverTool(Tool):
    name = "squad_retriever"
    description = """Retrieves documents from the Stanford Question Answering Dataset (SQuAD). 
        Because this tool does not remember context from previous queries, be sure to include any 
        relevant context in your query. Also, this tool only looks for affirmative matches, and does 
        not support negative queries, so only query for what you want, not what you don't want. 
        """
    inputs = {
        "query": {
            "type": "string",
            "description": "The query. This could be the literal question being asked by the user, modified to be informed by your goals and chat history. Be sure to pass this as a keyword argument and not a dictionary.",
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = Data()
        self.retriever = self.data.index.as_retriever()

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        responses = self.retriever.retrieve(query)

        if len(responses) == 0:
            return "No documents found for this query."
        return "===Document===\n" + "\n===Document===\n".join(
            [
                f"{response.text}\nScore: {response.score}"
                for response in responses
            ]
        )


class SquadQueryTool(Tool):
    name = "squad_query"
    description = """Attempts to answer a question using the Stanford Question Answering Dataset (SQuAD).
        Because this tool does not remember context from previous queries, be sure to include 
        any relevant context in your query."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The question. This should be the literal question being asked, only modified to be informed by your goals and chat history. Be sure to pass this as a keyword argument and not a dictionary.",
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = Data()
        self.query_engine = self.data.index.as_query_engine()

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        response = self.query_engine.query(query)
        # docs = self.data.index.similarity_search(query, k=3)

        if len(response.response) == 0:
            return "No answer found for this query."
        return "Query Response:\n\n" + "\n===Response===\n".join([response.response])
