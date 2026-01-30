# -*- coding: utf-8 -*-
"""
This file has been modified from the original Virtual-Scientists-v2-main project.

Original work: Virtual-Scientists-v2-main
Copyright: See LICENSE file in Virtual-Scientists-v2-main directory

This file is licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications:
- Added utility functions for different platform variants
- Added helper functions for real and social variants
"""

"""utils."""

from loguru import logger
from tqdm import tqdm
from collections import Counter
from typing import Union, Any, Sequence
import numpy as np
import json
import sqlite3
import re
import os
import sys
import difflib

sys.path.append('../camel-master')
# from agentscope.service import (
#     dblp_search_publications,  # or google_search,
#     arxiv_search
# )

from utils.prompt import Prompts

def majority_vote(votes: list) -> Any:
    """majority_vote function"""
    votes_valid = [item for item in votes if item != "Abstain"]
    # Count the votes excluding abstentions.
    unit, counts = np.unique(votes_valid, return_counts=True)
    return unit[np.argmax(counts)]


def extract_name_and_id(name: str) -> tuple[str, int]:
    """extract player name and id from a string"""
    try:
        name = re.search(r"\b[Pp]layer\d+\b", name).group(0)
        idx = int(re.search(r"[Pp]layer(\d+)", name).group(1)) - 1
    except AttributeError:
        # In case Player remains silent or speaks to abstain.
        logger.warning(f"vote: invalid name {name}, set to Abstain")
        name = "Abstain"
        idx = -1
    return name, idx


def extract_scientist_names(name: str) -> list:
    """extract player name and id from a string"""
    try:
        matches = re.findall(r"\b[S]cientist\d+\b", name)
        # idx = int(re.search(r"[Pp]layer(\d+)", name).group(1)) - 1
        names = [f"{num}" for num in matches]
    except AttributeError:
        # In case Player remains silent or speaks to abstain.
        logger.warning(f"vote: invalid name {name}, set to Abstain")
        names = ["Abstain"]
        idx = -1
    return list(set(names))


def team_description(team: list, over_state: int) -> str:
    """combine agent names into a string, and use "and" to connect the last
    two names."""
    output_string = "{"
    i = 1
    for team_index in team:
        if team_index.state != over_state:
            output_string += f"team{i}: {team_index.teammate}"
            i = i + 1
            if i < len(team):
                output_string += ", "
    output_string += "}"

    return output_string

def team_description_detail(team: list, agent_list: list, over_state: int) -> str:
    """combine agent names into a string, and use "and" to connect the last
    two names."""
    output_string = ""
    i=1
    for team_index in range(len(team)):
        if team[team_index].state!=over_state:
            team_list = team[team_index].teammate
            output_string += f"The Team{i} includes team members: {team_list}. "
            i=i+1
    output_string_before = f"You are currently a member of {i-1} teams. "
    # for agent in agent_list:
    #     if agent.name in independent_list:
    #         output_string += f"For {agent.name}, "
    #         output_string += convert_you_to_other(agent.sys_prompt)
    output_string = output_string_before + output_string + "Summarize the status of all the teams you are currently part of."
    return output_string

def convert_you_to_other(origin: str) -> str:
    after = origin.replace("Your", "His")
    after = after.replace("You", "He")
    after = after.replace("you", "he")
    after = after.replace("your", "his")
    return after

def read_txt_files_as_dict_continue_real(folder_path):
    dict_list = []  
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"): 
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r') as file:
                file_content = file.read()
                try:
                    file_dict_old = eval(file_content)
                    file_dict={}
                    file_dict['title']=file_dict_old['title']
                    if isinstance(file_dict_old['abstract'], tuple):
                        file_dict['abstract']=file_dict_old['abstract'][0]
                    else:
                        file_dict['abstract']=file_dict_old['abstract']
                    file_dict['year']=file_dict_old['year']
                    file_dict['id'] = file_dict_old['id']
                    file_dict['authors'] = file_dict_old['authors']
                    file_dict['discipline'] = file_dict_old['discipline']
                    file_dict['keywords'] = file_dict_old['keywords']
                except json.JSONDecodeError:
                    print(f"文件 {filename} 的内容不是有效的JSON格式，跳过该文件。")
                    continue

                dict_list.append(file_dict)
    dict_list.sort(key=lambda x: x['id'])
    return dict_list

def read_txt_files_as_dict_real(folder_path):
    dict_list = []  
    id = 0
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"): 
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r') as file:
                file_content = file.read()
                try:
                    file_dict_old = eval(file_content)
                    file_dict={}
                    file_dict['title']=file_dict_old['title']
                    if isinstance(file_dict_old['abstract'], tuple):
                        file_dict['abstract']=file_dict_old['abstract'][0]
                    else:
                        file_dict['abstract']=file_dict_old['abstract']
                    file_dict['year']=-1
                    file_dict['id'] = id
                    file_dict['authors'] = None
                    file_dict['discipline'] = None
                    file_dict['keywords'] = None
                except json.JSONDecodeError:
                    print(f"文件 {filename} 的内容不是有效的JSON格式，跳过该文件。")
                    continue

                dict_list.append(file_dict)
                id = id + 1
    return dict_list

def read_txt_files_as_dict(folder_path):
    dict_list = [] 
    id = 0
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"): 
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r') as file:
                file_content = file.read()
                try:
                    file_dict_old = eval(file_content)
                    file_dict={}
                    file_dict['title']=file_dict_old['title']
                    if isinstance(file_dict_old['abstract'], tuple):
                        file_dict['abstract']=file_dict_old['abstract'][0]
                    else:
                        file_dict['abstract']=file_dict_old['abstract']
                    file_dict['year']=-1
                    file_dict['citation']=int(file_dict_old['citation'])
                    file_dict['id'] = id
                    file_dict['authors'] = None
                    file_dict['cite_papers'] = None
                    file_dict['reviews'] = None
                    file_dict['discipline'] = None
                    file_dict['keywords'] = None
                except json.JSONDecodeError:
                    print(f"文件 {filename} 的内容不是有效的JSON格式，跳过该文件。")
                    continue

                dict_list.append(file_dict)
                id = id + 1
    return dict_list

def read_txt_files_as_dict_continue(folder_path):
    dict_list = []  
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):  
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r') as file:
                file_content = file.read()
                try:
                    file_dict_old = eval(file_content)
                    file_dict={}
                    file_dict['title']=file_dict_old['title']
                    if isinstance(file_dict_old['abstract'], tuple):
                        file_dict['abstract']=file_dict_old['abstract'][0]
                    else:
                        file_dict['abstract']=file_dict_old['abstract']
                    file_dict['year']=file_dict_old['year']
                    file_dict['citation']=int(file_dict_old['citation'])
                    file_dict['id'] = file_dict_old['id']
                    file_dict['authors'] = file_dict_old['authors']
                    file_dict['cite_papers'] = file_dict_old['cite_papers']
                    file_dict['reviews'] = file_dict_old['reviews']
                    file_dict['discipline'] = file_dict_old['discipline']
                    file_dict['keywords'] = file_dict_old['keywords']
                except json.JSONDecodeError:
                    print(f"文件 {filename} 的内容不是有效的JSON格式，跳过该文件。")
                    continue

                dict_list.append(file_dict)
    dict_list.sort(key=lambda x: x['id'])
    return dict_list

def write_txt_file_as_dict(file_path, file_dict, origin_len):
    for i in range(origin_len, len(file_dict)):
        filename = f"{file_path}/paper_{i}.txt"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(str(file_dict[i]))


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that some models prepend."""
    if text is None:
        return ''
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


def extract_between_json_tags(text, num=None):
    text = strip_think_tags(text)
    json_blocks = re.findall(r'```json(.*?)```', text, re.DOTALL)

    if not json_blocks:
        json_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if not json_blocks:
            json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if json_match:
                combined_json = json_match.group(0).strip()
            else:
                combined_json = text.strip()
            if not combined_json:
                combined_json = 'No response.'
            return combined_json
        else:
            if num==None:
                combined_json = ''.join(block.strip() for block in json_blocks)
            else:
                combined_json = ''.join(block.strip() for block in json_blocks[:num])
            if combined_json == None:
                combined_json = 'No response.'
            return combined_json
    else:
        if num==None:
            combined_json = ''.join(block.strip() for block in json_blocks)
        else:
            combined_json = ''.join(block.strip() for block in json_blocks[:num])
        if combined_json == None:
            combined_json = 'No response.'
        return combined_json


def extract_metrics(text, split_keywords):
    metrics = {}

    for keyword in split_keywords:
        parts = text.split(keyword)
        if len(parts) > 1:
            match = re.search(r'\d+', parts[1])
            if match:
                value = int(match.group())
                metrics[keyword.strip('"')] = value
            else:
                metrics[keyword.strip('"')] = None
        else:
            metrics[keyword.strip('"')] = None

    return metrics

def strip_non_letters(text):
    return re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', text)

def extract_after_think(text: str) -> str:
    import re
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
    
def save2database_real(paper_list : list[dict], output_dir : str):
    # connect to database, if it exists then delete it
    if os.path.isfile(output_dir):
        os.remove(output_dir)
    conn = sqlite3.connect(output_dir)
    # create cursor
    cursor = conn.cursor()

    # create table
    print('build paper table...')
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY,
                title TEXT,
                authors TEXT,
                abstract TEXT,
                year INTEGER,
                discipline TEXT,
                keyword TEXT
            )
        ''')

    for paper in paper_list:
        id = int(paper['id'])
        title = paper['title']
        abstract = paper['abstract']
        year = int(paper['year'])
        discipline = paper['discipline']
        keyword = paper['keywords']
        if paper['authors']!=None:
            authors = ';'.join(paper['authors'])
        else:
            authors = None

        # Define your user data (id, name, affiliation)
        paper_data = (id,
                      title,
                      authors,
                      abstract,
                      year,
                      discipline,
                      keyword)

        # Insert query
        query = '''
            INSERT INTO papers (id, title, authors, abstract, year,discipline, keyword)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            '''

        # Execute the query with user data
        cursor.execute(query, paper_data)
    
    print('build paper table successfully...')
    conn.commit()
    cursor.close()
    conn.close()
    
def save2database(paper_list : list[dict], output_dir : str):
    # connect to database, if it exists then delete it
    if os.path.isfile(output_dir):
        os.remove(output_dir)
    conn = sqlite3.connect(output_dir)
    # create cursor
    cursor = conn.cursor()

    # create table
    print('build paper table...')
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY,
                title TEXT,
                authors TEXT,
                cite_papers TEXT,
                abstract TEXT,
                year INTEGER,
                citation INTEGER,
                reviews TEXT,
                discipline TEXT,
                keyword TEXT
            )
        ''')

    for paper in paper_list:
        id = int(paper['id'])
        title = paper['title']
        abstract = paper['abstract']
        year = int(paper['year'])
        citation = int(paper['citation'])
        discipline = paper['discipline']
        keyword = paper['keywords']
        review_score = paper['reviews']
        if paper['authors']!=None:
            authors = ';'.join(paper['authors'])
            paper_references = ';'.join(map(str, paper['cite_papers']))
            review_score = ';'.join(map(str, review_score))
        else:
            authors = None
            paper_references = None

        # Define your user data (id, name, affiliation)
        paper_data = (id,
                      title,
                      authors,
                      paper_references,
                      abstract,
                      year,
                      citation,
                      review_score,
                      discipline,
                      keyword)

        # Insert query
        query = '''
            INSERT INTO papers (id, title, authors, cite_papers, abstract, year, citation, reviews, discipline, keyword)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''

        # Execute the query with user data
        cursor.execute(query, paper_data)
    
    print('build paper table successfully...')
    conn.commit()
    cursor.close()
    conn.close()

def count_team(team_list: list[dict], over_state: int):
    num = 0
    for team in team_list:
        if team.state<over_state:
            num = num+1
    return num


def top_three_indices(lst):
    sorted_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)

    top_three = [index for index, value in sorted_indices[:3]]

    return top_three

def extract_first_number(s):
    match = re.search(r'\d+', s)
    if match:
        return match.group()
    return None


def most_frequent_element(arr):
    try:
        count = Counter(arr)

        most_common_element = count.most_common(1)[0][0]
    except:
        most_common_element = 0

    return most_common_element

def read_txt_files_as_list(folder_path):
    dict_list = []  

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):  
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                file_content = file.read()
                dict_list.append(file_content)
    return dict_list

def process_author_text(input_text):
    name_start = input_text.find("Your name is")
    name_end = input_text.find(",", name_start)
    name = input_text[name_start + len("Your name is "):name_end]

    affiliations_start = input_text.find("following affiliations")
    affiliations_end = input_text.find("],", affiliations_start)
    affiliations = input_text[affiliations_start + len("following affiliations "):affiliations_end + 1]
    affiliations = affiliations.replace("['", "").replace("']", "").replace("', '", "; ")

    topics_start = input_text.find("following topics")
    topics_end = input_text.find("],", topics_start)
    topics = input_text[topics_start + len("following topics "):topics_end + 1]
    topics = topics.replace("['", "").replace("']", "").replace("', '", ", ")

    papers_start = input_text.find("published")
    papers_end = input_text.find("papers", papers_start)
    num_papers = input_text[papers_start + len("published "):papers_end].strip()

    citations_start = input_text.find("have", papers_end)
    citations_end = input_text.find("citations", citations_start)
    num_citations = input_text[citations_start + len("have "):citations_end].strip()

    collaborators_start = input_text.find("these individuals")
    collaborators_end = input_text.find("].", collaborators_start)
    collaborators = input_text[collaborators_start + len("these individuals "):collaborators_end + 1]
    collaborators = collaborators.replace("['", "").replace("']", "").replace("', '", ", ")

    output_text = (
        f"{name} is affiliated with the following institutions: {affiliations}. "
        f"Their research has focused on topics such as {topics}. "
        f"{name} has published {num_papers} papers and has received {num_citations} citations. "
        f"Previous collaborations include work with individuals such as {collaborators}."
    )
    return output_text

class Color:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

def filter_out_number_n_symbol(text):
    filtered_text = re.sub(r'[^\w\s,&]', '', text)  # Remove symbols
    filtered_text = ''.join([char for char in filtered_text if not char.isdigit()])  # Remove numbers
    filtered_text = filtered_text.strip()  # Remove leading/trailing whitespace
    return filtered_text

def find_best_match(target, options, cutoff = 0.0):
    # Find the best match
    best_match = difflib.get_close_matches(target, options, n=1, cutoff=cutoff)
    return best_match[0] if best_match else None