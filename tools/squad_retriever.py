from transformers.agents.tools import Tool
from data import Data

class SquadRetrieverTool(Tool):
    name = "squad_retriever"
    description = "Answers questions from the Stanford Question Answering Dataset (SQuAD)."
    inputs = {
        "query": {
            "type": "string",
            "description": "The question. This should be the literal question being asked, only modified to be informed by chat history. Be sure to pass this as a keyword argument and not a dictionary.",
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
        return "Retrieved answer:\n\n" + "\n===Answer===\n".join(
            [response.response]
        )