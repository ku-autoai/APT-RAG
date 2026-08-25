from utils.basic_prompt import RETRIEVAL_PROMPT, ROOT_SYS_PROMPT
from utils.utils import run_llm_with_cost


class Generator:
    def __init__(self, llm_type, tokenizer, model, dataset_name):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model
        self.dataset_name = dataset_name

    def LLM_Generate(self, query, g_cost, max_token):
        input_prompt = f"Question: {query}\nAnswer:"
        return run_llm_with_cost(
            llm_type=self.llm_type,
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=ROOT_SYS_PROMPT[self.dataset_name],
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )

    def Retrieval_Generate(self, query, documents, g_cost, max_token):
        input_prompt = f"Question: {query}\n\nDocuments:"
        for docs in documents:
            input_prompt += f"title: {docs['title']}\ncontent:{docs['content']}\n\n"
        input_prompt += RETRIEVAL_PROMPT
        input_prompt += "\nAnswer:"
        return run_llm_with_cost(
            llm_type=self.llm_type,
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=ROOT_SYS_PROMPT[self.dataset_name],
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )
