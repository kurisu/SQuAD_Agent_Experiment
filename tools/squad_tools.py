from transformers.agents.tools import Tool
from data import get_data

class SquadRetrieverTool(Tool):
    name = "squad_retriever"
    description = """Retrieves documents from the Stanford Question Answering Dataset (SQuAD). 
        Because this tool does not remember context from previous queries, be sure to include 
        as many details as possible in your query. 
        """
    inputs = {
        "query": {
            "type": "string",
            "description": "The query. Be sure to pass this as a keyword argument and not a dictionary.",
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = get_data(download=True)
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
        as many details as possible in your query."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The question. Be sure to pass this as a keyword argument and not a dictionary.",
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = get_data(download=True)
        self.query_engine = self.data.index.as_query_engine()

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        response = self.query_engine.query(query)

        if len(response.response) == 0:
            return "No answer found for this query."
        return "Query Response:\n\n" + "\n===Response===\n".join([response.response])
