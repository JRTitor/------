from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import json
from tqdm.notebook import tqdm

PATH_ENV = os.path.join(os.getcwd(), 'env.json')
api_data = json.load(open(PATH_ENV))

os.environ['LANGCHAIN_API_KEY'] = api_data['langchain_api']
os.environ['LANGCHAIN_TRACING_V2'] = 'True'

class message_dataset(Dataset):
    def __init__(self, path:str):
        self.messages = pd.read_csv(path, names=['message'])

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return self.messages.message.iloc[idx] 

def get_response(sys_prompt:str, complaint:str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', sys_prompt),
            ('user', 'Question:{question}')
        ]
    )

    ## Ollama tinyLlama LLM
    llm = Ollama(model='tinyLlama')
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    return chain.invoke({'question':complaint})

def prompt_runner(prompt:str, message_loader:DataLoader):
    return [(message[0], get_response(prompt, message[0])) for message in tqdm(message_loader)]


def main():
    prompts_df = pd.read_csv('../prompts/prompts.csv', names=['prompt'])
    message_path = '../messages/negative_mess.csv'
    message_data = message_dataset(message_path)
    message_loader = DataLoader(message_data)
    
    for i in tqdm(range(len(prompts_df))):
        prompt = prompts_df.prompt.iloc[i]
        tmp_list = prompt_runner(prompt, message_loader)
        df_to_save = pd.DataFrame({
                        "message":tmp_list[0],
                        "response":tmp_list[1],
                        })
        df_name = prompt.replace(' ', '_').replace('.', '0') + '.json'
        df_path = os.path.join('..\\joined', df_name)
        df_to_save.to_csv(df_path)


if __name__ == '__main__':
    main()
