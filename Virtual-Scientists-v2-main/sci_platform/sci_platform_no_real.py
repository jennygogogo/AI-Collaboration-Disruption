"""
This file is a new file based on the Virtual-Scientists-v2-main project.

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

This file is a variant of sci_platform_fast.py that:
- Uses vllm for model inference instead of ollama
- Removes reviewer functionality (no_real version)
- Adapted for simplified simulation scenarios without real paper review process
"""

import sys
import os
import numpy as np
import json
import re
from functools import partial
import faiss
from typing import Any
import random

sys.path.append('../camel-master')
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from social_agent.channel import Channel
from social_agent.sci_agent import SciAgent_Async

from sci_team.SciTeam_no_real import Team
from utils.prompt import Prompts
from utils.scientist_utils import (
    team_description,
    convert_you_to_other,
    team_description_detail,
    read_txt_files_as_dict_real,
    read_txt_files_as_dict_continue_real,
    extract_between_json_tags,
    count_team,
    write_txt_file_as_dict,
    save2database_real,
    extract_after_think
)

import asyncio
from inference.inference_manager_vllm import InferencerManager

class Platform:
    r"""Platform."""

    def __init__(self,
                 agent_num: int = 1,
                 ips: list = ['127.0.0.1'],
                 port: list = [11434],
                 model_name: str='qwen3:8b',
                 author_folder_path: str = '/inspire/ssd/project/aiscientist/hufang-CZXS24220079/Virtual-Scientists-main/Data/Authors/books_OAG_10000_after',
                 paper_folder_path: str = '/inspire/ssd/project/aiscientist/hufang-CZXS24220079/Virtual-Scientists-main/Data/Papers/papers_OAG',
                 future_paper_folder_path: str = '/inspire/ssd/project/aiscientist/hufang-CZXS24220079/Virtual-Scientists-main/Data/Papers/papers_future_OAG',
                 adjacency_matrix_dir: str = '/inspire/ssd/project/aiscientist/hufang-CZXS24220079/Virtual-Scientists-main/Data',
                 paper_index_path: str='/inspire/ssd/project/aiscientist/hufang-CZXS24220079/Virtual-Scientists-main/Data/Embeddings/faiss_index_OAG.index',

                 paper_future_index_path: str='/inspire/ssd/project/aiscientist/hufang-CZXS24220079/Virtual-Scientists-main/Data/Embeddings/faiss_index_OAG_future.index',
                 checkpoint_path: str='./database',
                 log_dir: str = 'logs',
                 info_dir: str = "team_info",
                 group_max_discuss_iteration: int = 2, # 6， 7
                 recent_n_team_mem_for_retrieve: int = 3,
                 recent_n_agent_mem_for_retrieve: int = 1,
                 team_limit: int = 2,
                 check_iter: int = 5,
                 review_num: int = 2,
                 max_teammember: int = 3,
                 default_mark: int = 4,
                 skip_check: bool = False,
                 over_state: int = 7,
                 begin_state: int = 1,
                 explore: str = 'gaussian', # 'uniform' or 'gaussian' or 'history'
                 team_organization: str = 'exponential', # 'uniform' or 'gaussian' or 'exponential'
                 checkpoint: bool = True,
                 test_time: str = 'None',
                 load_time: str = 'None',
                 leader_mode: str = 'normal', # 'normal' or 'random'
                 ):

        self.agent_num = agent_num
        self.port = port
        self.ips = ips
        self.paper_folder_path = paper_folder_path
        self.paper_future_folder_path = future_paper_folder_path
        self.author_info_dir = os.path.join(author_folder_path,'author_{}.txt')
        self.adjacency_matrix_dir = adjacency_matrix_dir
        self.group_max_discuss_iteration = group_max_discuss_iteration
        self.recent_n_team_mem_for_retrieve = recent_n_team_mem_for_retrieve
        self.recent_n_agent_mem_for_retrieve = recent_n_agent_mem_for_retrieve
        # how many teams for one agent is allowed
        self.team_limit = team_limit
        # how many times to try paper search
        self.check_iter = check_iter
        # the max team member in a team
        self.max_teammember = max_teammember
        # default review mark
        self.default_mark = default_mark
        # check novelty
        self.skip_check = skip_check
        # current state for the over of team activity
        self.over_state = over_state
        # current state for the begin of team activity
        self.begin_state = begin_state
        # output dir
        self.log_dir = log_dir
        self.info_dir = info_dir
        self.author_folder_path = author_folder_path
        self.checkpoint_path = checkpoint_path
        self.explore = explore
        self.team_organization = team_organization
        self.unactivation = 0
        self.leader_mode = leader_mode

        # for quality, the team of one member will think more times
        self.think_times = max_teammember+1

        # load k-hop adjacency matrix
        self.degree_int2word = ['one', 'two', 'three', 'four', 'five']

        self.adjacency_matrix = np.loadtxt(
            '{}/weight_matrix.txt'.format(self.adjacency_matrix_dir), dtype=float)
        
        self.checkpoint = checkpoint
        self.test_time = test_time
        self.load_time = load_time
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        folder_path = f"{checkpoint_path}/{self.test_time}"

        if os.path.exists(folder_path):
            os.system(f"rm -rf {folder_path}")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            subfolders = ["paper", "faiss", "team","citation"]
            for sub in subfolders:
                sub_path = os.path.join(folder_path, sub)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)

        # dict for paper citation
        self.paper_citation_list = {}

        # check if agent_num is valid
        if self.agent_num is None:
            self.agent_num = len(self.adjacency_matrix)
        else:
            assert self.agent_num <= len(self.adjacency_matrix)

        self.inference_channel = Channel()
        self.embed_inference_channel = Channel()

        concurrency_level = 128 
        vllm_ports = [8000] * concurrency_level
        
        ollama_embed_ports = [11434, 11435, 11436,11437,11438,11439,11440]
        # ollama_embed_ports = [11461, 11462, 11463]

        self.inference_configs = {
            'model_type': model_name,
            'embed_model_type': None,  
            'model_path': 'API',
            'stop_tokens': None,
            'server_url': [{'host': '127.0.0.1', 'ports': vllm_ports}]
        }
        
        self.embed_inference_configs = {
            'model_type': model_name,
            'embed_model_type': "mxbai-embed-large",
            'model_path': 'API',
            'stop_tokens': None,
            'server_url': [{'host': '127.0.0.1', 'ports': ollama_embed_ports}]
        }
        self.infere = InferencerManager(
            self.inference_channel,
            **self.inference_configs,
        )
        self.embed_infere = InferencerManager(
            self.embed_inference_channel,
            **self.embed_inference_configs,
        )

        self.agent_pool = self.init_agent_async(self.inference_channel, self.embed_inference_channel, self.author_info_dir, len(self.adjacency_matrix), model_name)

        self.id2agent = {}
        for agent in self.agent_pool:
            self.id2agent[agent.role_name] = agent
        if not self.checkpoint:
            # team pool
            self.team_pool = []
            self.team_count = []
            agent_id = 1
            random_indices = random.sample(range(len(self.agent_pool)), self.agent_num)
            self.random_indices = random_indices
            for i in random_indices:
                agent = self.agent_pool[i]
                team_agent = []
                team_index = []
                team_index.append(agent.role_name)
                team_dic = Team(team_name = str(agent_id)+','+str(1),
                                log_dir = self.log_dir,
                                info_dir = self.info_dir,
                                recent_n_team_mem_for_retrieve = self.recent_n_team_mem_for_retrieve)
                team_dic.teammate = team_index
                team_agent.append(team_dic)
                self.team_pool.append(team_agent)
                self.team_count.append(1)
                agent_id = agent_id + 1
        else:
            with open(f"{checkpoint_path}/{self.load_time}/team/team_preamble.txt", 'r') as file:
                file_content = file.read()
                dict = eval(file_content)
                self.team_count=dict['team_count']
                self.random_indices=dict['team_indices']
            self.team_pool=[]
            for team_leader in range(len(self.random_indices)):
                with open(f"{checkpoint_path}/{self.load_time}/team/{team_leader}.txt", 'r') as file:
                    team_leader_list = []
                    for line in file:
                        team_index = json.loads(line.strip())
                        team_leader_list.append(Team.load_from_file(team_index))
                    self.team_pool.append(team_leader_list)

        cpu_index = faiss.read_index(paper_index_path)

        res = faiss.StandardGpuResources() 
        self.gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  

        cpu_future_index = faiss.read_index(paper_future_index_path)

        future_res = faiss.StandardGpuResources() 
        self.gpu_future_index = faiss.index_cpu_to_gpu(future_res, 0, cpu_future_index) 

        self.paper_dicts = read_txt_files_as_dict_real(self.paper_folder_path)
        self.epoch = 0
        self.origin_len = len(self.paper_dicts)
        if self.checkpoint:
            self.continue_paper_folder_path = f"{checkpoint_path}/{self.load_time}/paper"
            self.continue_folder_path = f"{checkpoint_path}/{self.load_time}/faiss"
            self.continue_paper_dicts = read_txt_files_as_dict_continue_real(self.continue_paper_folder_path)
            self.paper_dicts = self.paper_dicts+self.continue_paper_dicts

            cpu_index = faiss.read_index(os.path.join(self.continue_folder_path, 'faiss_index_OAG.index'))
            res = faiss.StandardGpuResources() 
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index) 
            self.epoch = self.paper_dicts[-1]['year']+1
        print(self.origin_len)
        print(len(self.paper_dicts))
        print(self.gpu_index.ntotal)
    

    def init_agent_async(self, channel, embed_channel, information_path, count, model_name):
        agents = []
        inference_channel = channel

        # 创建并注册用户
        for i in range(count):
            information_path_index = information_path.format(i)
            with open(information_path_index, 'r') as file:
                prompt = file.read()
            name = 'Scientist{}'.format(i)
            prompt = BaseMessage.make_assistant_message(
                role_name=name,
                content=prompt,
            )
            agent = SciAgent_Async(prompt, channel=inference_channel, embed_channel=embed_channel, token_limit=32768, message_window_size = self.recent_n_agent_mem_for_retrieve)
            # 设置模型类型为qwen，覆盖默认的llama配置
            agent.model_type = model_name
            agents.append(agent)

        return agents

    async def select_single(self, agent_index):
        scientists = [self.agent_pool[i] for i in self.random_indices]
        if len(self.team_pool[agent_index]) > 0:
            if self.team_pool[agent_index][0].state>=2:
                return
        else:
            return
        # avoid too many teams
        if count_team(self.team_pool[agent_index], self.over_state)>=self.team_limit:
            return
        sys_prompt = scientists[agent_index].orig_sys_message.content + Prompts.role
        hint = BaseMessage.make_user_message(role_name="user",
                                             content=Prompts.ask_choice.format_map(
                                                 {"Scientist_name": scientists[agent_index].role_name,
                                                  "All_team": team_description(self.team_pool[agent_index],
                                                                               self.over_state)})
                                             )
        x = await scientists[agent_index].step(hint)
        x= x.msg
        x.content = extract_after_think(x.content)
        self.team_pool[agent_index][0].log_dialogue('user', hint.content)
        self.team_pool[agent_index][0].log_dialogue(scientists[agent_index].role_name, x.content)
        match = re.search(r'(\d+)', extract_between_json_tags(x.content), re.IGNORECASE)
        if match != None:
            # when action2, the agent choose to act independently
            if int(match.group(1))==2:
                print("Single Agent Independently!")
                self.team_pool[agent_index][0].state=2
                return

        # use prompts to select scientists
        scientist = scientists[agent_index].role_name
        name = int(scientist[9:])
        arr = self.adjacency_matrix[name, :].copy()
        # uniform distribution
        if self.explore == 'uniform':
            arr += 1
        # sample from gaussian distribution
        elif self.explore == 'gaussian':
            random_values = np.random.normal(loc=0.005, scale=0.005, size=arr.shape)
            random_values = np.abs(random_values)
            random_values[random_values > 0.01] = 0.01
            arr += random_values
        else:
            # extract the index that is not 0, 2-jump
            arr_2 = np.nonzero(arr)[0]
            for i in arr_2:
                for j in range(len(self.adjacency_matrix)):
                    arr[j] = arr[j] + 0.3*self.adjacency_matrix[i, j]
        arr[agent_index] = 0
        probabilities = arr / np.sum(arr)

        if self.team_organization == 'uniform':
            team_size = self.max_teammember
        elif self.team_organization == 'gaussian':
            # team member follows the distribution
            team_sample = np.random.normal(loc=self.max_teammember, scale=1)
            team_sample_int = int(np.round(team_sample))
            team_size = np.clip(team_sample_int, 3, 2*self.max_teammember-3)
        else:
            lambda_val = 0.25  
            scale = 1 / lambda_val  
            team_sample = np.random.exponential(scale, 1)+3
            team_sample_int = np.floor(team_sample).astype(int)
            team_size = np.clip(team_sample_int, 3, 2*self.max_teammember+3)
        
        selected_indices = np.random.choice(len(arr), size=team_size, p=probabilities, replace=False)

        team_candidate = []
        for i in range(len(selected_indices)):
            team_candidate.append(f"Scientist{selected_indices[i]}")

        # print(team_candidate)
        self.team_pool[agent_index][0].log_dialogue(scientists[agent_index].role_name, ','.join(team_candidate))

        # ask each scientist to decide whether to join
        agent_candidate = self.id_to_agent(team_candidate)
        # create new team
        team_index = []
        team_index.append(scientists[agent_index].role_name)
        for agent in agent_candidate:
            if agent.role_name == scientists[agent_index].role_name:
                continue

            hint = BaseMessage.make_user_message(content=Prompts.to_scientist_choice.format_map({
                "inviter_name": scientists[agent_index].role_name,
                "team_member": str(team_index),
                "personal information" : convert_you_to_other(sys_prompt)
            }), role_name="User")
            # set_parsers(agent, Prompts.scientist_invite_parser)
            pattern = re.compile(r'1', re.IGNORECASE)
            # action1 means a scientist accepts the invitance
            x = await agent.step(hint)
            x = x.msg
            x.content = extract_after_think(x.content)
            if pattern.search(extract_between_json_tags(x.content, num=1)):
                team_index.append(agent.role_name)
            # self.team_pool[agent_index][0].log_dialogue('user', hint.content)
            # self.team_pool[agent_index][0].log_dialogue(agent.role_name, x.content)
        team_count = self.team_count[agent_index]+1
        team_dic = Team(team_name = str(agent_index+1)+','+str(team_count),
                        log_dir = self.log_dir,
                        info_dir = self.info_dir,
                        recent_n_team_mem_for_retrieve = self.recent_n_team_mem_for_retrieve)
        team_dic.state=2
        team_dic.teammate = team_index
        self.team_pool[agent_index].append(team_dic)
        self.team_count[agent_index]=team_count

        # connetion between collaborators will be closer
        for member in team_dic.teammate:
            if int(member[9:])!=self.random_indices[agent_index]:
                self.adjacency_matrix[self.random_indices[agent_index], int(member[9:])]=self.adjacency_matrix[self.random_indices[agent_index], int(member[9:])]+1
                self.adjacency_matrix[int(member[9:]), self.random_indices[agent_index]]=self.adjacency_matrix[int(member[9:]), self.random_indices[agent_index]]+1
        # summary current teams in memory
        summary_select = await scientists[agent_index].step(BaseMessage.make_user_message(
            content=team_description_detail(self.team_pool[agent_index], self.agent_pool, self.over_state),
            role_name="User"))
        summary_select.msg.content = extract_after_think(summary_select.msg.content)
        self.team_pool[agent_index][0].log_dialogue(scientists[agent_index].role_name, summary_select.msg.content)

    async def select_coauthors(self,):
        # decide whether the scientist wants to find partners
        select_tasks = []
        for agent_index in range(self.agent_num):
            select_tasks.append(self.select_single(agent_index))
        await asyncio.gather(*select_tasks) 
        team_list = self.team_pool
        return team_list

    def id_to_agent(self, team_list):
        agent_list = []
        for agent_id in team_list:
            agent_list.append(self.id2agent[agent_id])
        return agent_list

    def agent_to_id(self, team_list):
        agent_list = []
        for agent_id in team_list:
            agent_list.append(agent_id.role_name)
        return agent_list
    

    async def team_running(self, epoch, leader_index):
        # leader_team=[]
        for team_index in range(len(self.team_pool[leader_index])):
            self.team_pool[leader_index][team_index].epoch = epoch
            await self.team_pool[leader_index][team_index].action_excution(self)

    async def running(self, epochs):
        # 创建调度任务
        self.inference_task = asyncio.create_task(self.infere.run())
        self.embed_inference_task = asyncio.create_task(self.embed_infere.run())
        # init team_pool
        print(f'{"="*50}Epoch:{-1} | Initialize Teams {"="*50}')
        self.team_pool = await self.select_coauthors()

        for epoch in range(self.epoch, epochs):
            # state 7 is an over
            # 1. select coauthors for state 1
            # 2. select topics for state 2
            # 3. generate idea for state 3Si
            # 4. check novelty for state 4
            # 5. generate paper abstract for state 5
            # 6. generate paper review for state 6
            leader_tasks = []
            for leader_index in range(len(self.team_pool)):
                leader_tasks.append(self.team_running(epoch, leader_index))

            await asyncio.gather(*leader_tasks) 

            print(f'{"="*50} Epoch:{epoch} | Begin Select Authors {"="*50}')
            self.team_pool = await self.select_coauthors()
            print(f'{"="*50} Epoch:{epoch} | Current Action Finished {"="*50}')
            
            # save database for statistics
            output_dir = f"{self.checkpoint_path}/{self.test_time}/database_large.db"
            save2database_real(self.paper_dicts, output_dir)

            # save faiss for similarity
            temp_index = faiss.index_gpu_to_cpu(self.gpu_index)
            file_path = f"{self.checkpoint_path}/{self.test_time}/paper"
            faiss.write_index(temp_index, f"{self.checkpoint_path}/{self.test_time}/faiss/faiss_index_OAG.index")
            
            # save txt for paper
            write_txt_file_as_dict(file_path, self.paper_dicts, self.origin_len)
            team_information_dict = {
                'team_count': self.team_count,
                'team_indices': self.random_indices
            }
            with open(f"{self.checkpoint_path}/{self.test_time}/team/team_preamble.txt", "w", encoding="utf-8") as f:
                f.write(str(team_information_dict))
            for team_leader in range(len(self.random_indices)):
                path = f"{self.checkpoint_path}/{self.test_time}/team/{team_leader}.txt"
                if os.path.exists(path):
                    os.remove(path)
                for team_index in self.team_pool[team_leader]:
                    team_index.save_to_file(path)
            

        await self.infere.stop()
        await self.embed_infere.stop()
        await self.inference_task
        await self.embed_inference_task

        # save self.adjacency_matrix
        np.savetxt(f'{self.checkpoint_path}/{self.test_time}/weight_matrix.txt', self.adjacency_matrix, fmt='%d')
    