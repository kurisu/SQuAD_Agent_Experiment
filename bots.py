from data import Data

'''
The BotWrapper class makes it so that different types of bots can be used in the same way.
This is used in the Bots class to create a list of all bots and pass them to the frontend.
'''
class BotWrapper:
    def __init__(self, bot):
        self.bot = bot

    def chat(self, *args, **kwargs):
        methods = ['chat', 'query']
        for method in methods:
            if hasattr(self.bot, method):
                print(f"Calling {method} method")
                method_to_call = getattr(self.bot, method)
                return method_to_call(*args, **kwargs).response()
        raise AttributeError(f"'{self.bot.__class__.__name__}' object has none of the required methods: '{methods}'")  
     
    def stream_chat(self, *args, **kwargs):
        methods = ['stream_chat', 'query']
        for method in methods:
            if hasattr(self.bot, method):
                print(f"Calling {method} method")
                method_to_call = getattr(self.bot, method)
                return method_to_call(*args, **kwargs).response_gen
        raise AttributeError(f"'{self.bot.__class__.__name__}' object has none of the required methods: '{methods}'")   

'''
The Bots class creates the bots and passes them to the frontend.
'''
class Bots:
    def __init__(self):
        self.data = Data()
        self.data.load_data()
        self.query_engine = None
        self.chat_agent = None
        self.all_bots = None
        self.create_bots()

    def create_query_engine_bot(self):
        if self.query_engine is None:
            self.query_engine = BotWrapper(self.data.index.as_query_engine())
        return self.query_engine
    
    def create_chat_agent(self):
        if self.chat_agent is None:
            from llama_index.core.memory import ChatMemoryBuffer
            memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
            self.chat_agent = BotWrapper(self.data.index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                context_prompt=(
                    "You are a chatbot, able to have normal interactions, as well as talk"
                    " about the questions and answers you know about."
                    "Here are the relevant documents for the context:\n"
                    "{context_str}"
                    "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
                )
            ))
        return self.chat_agent

    def create_bots(self):
        self.create_query_engine_bot()
        self.create_chat_agent()
        self.all_bots = [self.query_engine, self.chat_agent]
        return self.all_bots
    
    def get_bots(self):
        return self.all_bots
