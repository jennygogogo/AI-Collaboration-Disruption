from datetime import datetime
from copy import deepcopy
import logging
import re
import ollama
import torch
import torch.nn.functional
import numpy as np
import json
import os
import sys
import random
import heapq
import asyncio
sys.path.append('../camel-master')

from camel.messages import BaseMessage
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelPlatformType,
    ModelType,
    OpenAIBackendRole,
    RoleType,
)

from utils.prompt import Prompts
from utils.scientist_utils import (
    extract_scientist_names,
    team_description,
    convert_you_to_other,
    team_description_detail,
    read_txt_files_as_dict,
    extract_between_json_tags,
    extract_metrics,
    strip_non_letters,
    save2database,
    count_team,
    top_three_indices,
    extract_first_number,
    most_frequent_element,
    Color,
    find_best_match,
    filter_out_number_n_symbol
)

class Team:
    def __init__(self, team_name, log_dir, info_dir, recent_n_team_mem_for_retrieve, state=1, epoch=-1, teammate=[], memory=[], mark_history=[],topic="None", idea="None", abstract="None",citation_id=[],self_review="None", paper_review="None"):
        # attrs
        self.team_name = team_name
        self.state = state
        self.epoch = epoch
        self.teammate = deepcopy(teammate)
        self.memory = deepcopy(memory)
        self.mark_history = deepcopy(mark_history)
        self.recent_n_team_mem_for_retrieve = recent_n_team_mem_for_retrieve
        self.topic = topic
        self.idea = idea
        self.abstract = abstract
        self.citation_id = deepcopy(citation_id)
        self.self_review = self_review
        self.paper_review = paper_review
        self.log_dir = log_dir
        self.info_dir = info_dir

        # state log
        self.state_log = {
            1: 'WAIT',
            2: 'TOPIC',
            3: 'IDEA',
            4: 'CHECK',
            5: 'ABSTRACT',
            6: 'REVIEW',
            7: 'FINISH'
        }

        # state action
        self.state_action = {
            2: self.select_topic,
            3: self.generate_idea,
            4: self.check_novelty,
            5: self.generate_abstract,
            6: self.generate_review
        }

        # init log file dir
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.info_file = f"{info_dir}/{current_time}_{self.team_name}_dialogue.json"
        # self.log_file = f"{log_dir}/{current_time}_{self.team_name}_dialogue.log"
        # self.log_file_all = f"{log_dir}/ALL_dialogue.log"

        # Check if log file exists and delete it
        # if os.path.exists(self.log_file):
        #     os.remove(self.log_file)

        # self.logger = logging.getLogger(self.team_name)
        # self.logger_2 = logging.getLogger("ALL")
        # self.logger.setLevel(logging.INFO)
        # self.logger_2.setLevel(logging.INFO)
        # fh = logging.FileHandler(self.log_file)
        # fh_2 = logging.FileHandler(self.log_file_all)
        # self.logger.addHandler(fh)
        # self.logger_2.addHandler(fh_2)

    async def _safe_step(self, agent, prompt, timeout: float = 300.0, context: str = ""):
        """包装 agent.step，增加超时和异常兜底，避免单次调用卡死整个流程。"""
        name = getattr(agent, "role_name", "agent")
        try:
            resp = await asyncio.wait_for(agent.step(prompt), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"[WARN] step timeout ({context}) for {name}")
            return None
        except Exception as e:
            print(f"[WARN] step error ({context}) for {name}: {e}")
            return None
        if resp is None:
            return None
        return resp.msg

    async def _safe_embed(self, agent, prompt, timeout: float = 300.0, context: str = ""):
        """包装 agent.embed_step，增加超时和异常兜底，避免单次 embedding 卡死。"""
        name = getattr(agent, "role_name", "agent")
        try:
            resp = await asyncio.wait_for(agent.embed_step(prompt), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"[WARN] embed timeout ({context}) for {name}")
            return None
        except Exception as e:
            print(f"[WARN] embed error ({context}) for {name}: {e}")
            return None
        return resp

    # format memories
    def format_memories(self, current_memories = None, previous_memories = None, team_memories = None):
        memory_type_hint = ['Discussion in this turn', 'Summarization of previous turns', 'Team memory']

        output = ''

        if current_memories is not None and len(current_memories) != 0:
            format_current_memories = f'{memory_type_hint[0]}:\n'
            for memory in current_memories:
                format_current_memories += (memory.role_name + ': ' + memory.content + '\n')

            output = format_current_memories + output

        if previous_memories is not None and len(previous_memories) != 0:
            format_previous_memories = f'{memory_type_hint[1]}:\n'
            for memory in previous_memories:
                format_previous_memories += (memory.role_name + ': ' + memory.content + '\n')

            output = format_previous_memories + output

        if team_memories is not None and len(team_memories) != 0:
            format_team_memories = f'{memory_type_hint[2]}:\n'
            for memory in team_memories[-self.recent_n_team_mem_for_retrieve:]:
                format_team_memories += (memory.content + '\n')

            output = format_team_memories + output

        return output


    # execute action based on team state
    async def action_excution(self, platform):
        if self.state in self.state_log.keys():
            print(f'{"="*50} Epoch: {self.epoch} | BEGIN {self.state_log[self.state]} PROCESS {"="*50}')

        action = self.state_action.get(self.state, None)
        if action is not None:
            await action(platform)
            # print(f'{"="*50} Epoch: {self.epoch} | FINISH {self.state_log[self.state]} PROCESS {"="*50}')

    # general group discussion process
    async def group_discuss(self, platform, prompt: str = None):
        # prompt is used to start and guide the discussion
        # for each turn, in group_discuss, all dialogue history is stored in dialogue_history but not in agent memory
        # after finishing each discussion turn, agent1 will summarize dialogue_history and add a summarization into team_history

        # get teammate
        teammate = platform.id_to_agent(self.teammate)

        # current topic hot
        # get the 5 highest citation papers from old_hot
        if len(platform.old_hot) > 5:
            top_5_keys = heapq.nlargest(5, platform.old_hot, key=platform.old_hot.get)
            hot_content = ""
            for key in top_5_keys:
                hot_content += f"Paper {key}: {platform.paper_dicts[key]['abstract']}\n"

            hot_topic_prompt = Prompts.to_ask_hot_topic.replace("[hot topic]", hot_content)
            hot_topic_prompt = BaseMessage.make_user_message(role_name="user", content=hot_topic_prompt)
            try:
                random_summarizer = random.choice(teammate)
                hot_topic_reply = await random_summarizer.step(hot_topic_prompt)
                hot_topic_reply = hot_topic_reply.msg.content
                hot_topic = extract_between_json_tags(hot_topic_reply, num=1)
                hot_topic = strip_non_letters(hot_topic.split("Current Hotspots")[1])
                # print("Have hot topic")
                # print(f"-----------------hot topic:{hot_topic}-------------------")
            except:
                hot_topic = "No hot topic"
                # print('No response_hot_topic')
        else:
            hot_topic = "No hot topic"

        # Memoryies
        # get team_history, previous group discussion summarizations
        team_memories = self.memory
        # init previous_memories, a list of summarizations of previous turns in this group discussion
        previous_memories = []

        # init exit state
        exit = False
        # output return dialogue history, summarization of the last turn, and memory of the last turn
        output = {}
        # start discussing
        if len(teammate) == 1:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        else:
            group_max_discuss_iteration = platform.group_max_discuss_iteration

        for turn in range(group_max_discuss_iteration):
            # init current_memories for each turn
            current_memories = []

            said = []
            agent_num = 0
            for agent in teammate:
                if agent.role_name in said:
                    continue
                else:
                    said.append(agent.role_name)
                if len(hot_topic) > 50:
                    agent_prompt = f"Current team members are {','.join(self.teammate)}.\n" + \
                                f"Current hot topic is {hot_topic}.\n" + \
                                self.format_memories(None, None, team_memories) + \
                                prompt + \
                                self.format_memories(current_memories, previous_memories, None)
                else:
                    agent_prompt = f"Current team members are {','.join(self.teammate)}.\n" + \
                                self.format_memories(None, None, team_memories) + \
                                prompt + \
                                self.format_memories(current_memories, previous_memories, None)
                format_agent_prompt = BaseMessage.make_user_message(role_name="user", content=agent_prompt)

                # add reply to turn_history
                reply = await self._safe_step(agent,
                                              format_agent_prompt,
                                              timeout=6000.0,
                                              context="group_discuss")
                if reply is None:
                    print('No response_discussion')
                    fake_content = f"(No response from {agent.role_name} in this turn.)"
                    self.log_dialogue(agent.role_name, fake_content)
                    continue  
                else:
                    self.log_dialogue(agent.role_name, reply.content)

                current_memories.append(reply)
                agent_num = agent_num + 1
                # discussion is finished
                if 'exit' in reply.content:
                    exit = True
                    break

            # summarize this turn's discussion
            turn_summarization_prompt = 'Briefly summarize "Discussion in this turn".' +\
                                        self.format_memories(current_memories, previous_memories, None)
            format_turn_summarization_prompt = BaseMessage.make_user_message(role_name="user", content=turn_summarization_prompt)

            random_summarizer_turn = random.choice(teammate)
            x = await random_summarizer_turn.step(format_turn_summarization_prompt)
            x = x.msg
            self.log_dialogue(x.role_name, x.content)
            turn_summarization = BaseMessage.make_user_message(role_name="Summarization of turn{}".format(turn+1), content=x.content)

            if exit or turn==group_max_discuss_iteration-1:
                output['last_turn_summarization'] = turn_summarization
                output['last_turn_history'] = current_memories
                break
            else:
                # print(turn_summarization)
                previous_memories.append(turn_summarization)

        output['previous_memories'] = previous_memories
        self.teammate = platform.agent_to_id(teammate)
        return output

    async def select_topic(self, platform):
        # prompt to start discussing select_topic
        discuss_result = await self.group_discuss(platform, Prompts.to_start_topic_discussion)
        team_memories = self.memory
        previous_memories = discuss_result['previous_memories']
        current_memories = discuss_result['last_turn_history']
        last_turn_summarization = discuss_result['last_turn_summarization']

        answer_prompt = self.format_memories(current_memories, previous_memories, team_memories) +\
                        Prompts.to_ask_if_ready_give_topic
        format_answer_prompt = BaseMessage.make_user_message(role_name="user", content=answer_prompt)

        random_agent_id = random.randint(0, len(self.teammate)-1)
        answer_msg = await self._safe_step(
            platform.id2agent[self.teammate[random_agent_id]],
            format_answer_prompt,
            timeout=6000.0,
            context="select_topic_answer",
        )
        if answer_msg is None:
            answer_content = ""
        else:
            answer_content = answer_msg.content
        # self.log_dialogue('user', answer_prompt)
        self.log_dialogue(platform.id2agent[self.teammate[random_agent_id]].role_name, answer_content)
        answer_pattern = re.compile(r'1', re.IGNORECASE)

        # check whether agent is ready to answer
        if answer_pattern.search(answer_content) or len(team_memories)>=1:
            self.state = 3
            # prompt
            topic_prompt = Prompts.to_ask_topic.replace("[history_prompt]", self.format_memories(current_memories, previous_memories, team_memories))
            format_topic_prompt = BaseMessage.make_user_message(role_name="user", content=topic_prompt)
            # answer
            leader_index = random.randint(0, len(self.teammate)-1)
            topic_msg = await self._safe_step(
                platform.id2agent[self.teammate[leader_index]],
                format_topic_prompt,
                timeout=6000.0,
                context="select_topic_topic",
            )
            if topic_msg is None:
                self.topic = "No topic."
                previous_memories.append(last_turn_summarization)
                self.memory = team_memories
                return
            self.log_dialogue(platform.id2agent[self.teammate[leader_index]].role_name, topic_msg.content)
            try:
                self.topic = extract_between_json_tags(topic_msg.content, num=1)
                self.topic = strip_non_letters(self.topic.split("Selected Topic")[1])
                if len(self.topic)<3:
                    self.topic = topic_msg.content
                    self.topic = strip_non_letters(self.topic.split("Selected Topic")[1])
            except:
                self.topic = topic_msg.content
            # update dialogue history
            previous_memories.append(last_turn_summarization)
            topic_message = BaseMessage.make_user_message(role_name="user", content="Final selected topic: "+self.topic)
            previous_memories.append(topic_message)
        else:
            # update dialogue history
            previous_memories.append(last_turn_summarization)
        # summarize dialogue history
        dialogue_summarization_prompt = 'Briefly summarize "Summarizations of previous turns".' + \
                                        self.format_memories(None, previous_memories, team_memories)
        format_dialogue_summarization_prompt = BaseMessage.make_user_message(role_name="user", content=dialogue_summarization_prompt)
        random_agent_id_2 = random.choice(self.teammate)
        dialogue_msg = await self._safe_step(
            platform.id2agent[random_agent_id_2],
            format_dialogue_summarization_prompt,
            timeout=6000.0,
            context="select_topic_summary",
        )
        if dialogue_msg is not None:
            team_memories.append(dialogue_msg)
        self.memory = team_memories

    async def generate_idea(self, platform):
        topic = self.topic
        self.memory = []
        old_idea = None
        best_idea = None
        idea_list = []
        mark_list = []
        previous_memories = []
        # search related paper about the topic
        selected_topics = strip_non_letters(topic.split("Selected Topics:")[-1])

        teammate = platform.id_to_agent(self.teammate)
        idea_judge = True

        if len(teammate)==1:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        else:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        
        try:
            selected_topics_prompt = BaseMessage.make_user_message(role_name="user", content=selected_topics)
            resp = await self._safe_embed(
                teammate[0],
                selected_topics_prompt,
                timeout=1500.0,
                context="generate_idea_selected_topics",
            )
            if resp is None or getattr(resp, "data", None) is None:
                raise ValueError("Embedding failed for selected_topics.")
            query_vector = np.array([resp.data[0].embedding])
        except:
            selected_topics = "No topics."
            selected_topics_prompt = BaseMessage.make_user_message(role_name="user", content=selected_topics)
            resp = await self._safe_embed(
                teammate[0],
                selected_topics_prompt,
                timeout=1500.0,
                context="generate_idea_selected_topics_fallback",
            )
            if resp is None or getattr(resp, "data", None) is None:
                query_vector = np.random.rand(1, 1024)
            else:
                query_vector = np.array([resp.data[0].embedding])
        paper_reference, cite_paper = await platform.reference_paper_alignment(query_vector, platform.cite_number, self.epoch)

        for turn in range(group_max_discuss_iteration):
            current_memories = []
            # discuss the idea
            for agent in teammate:
                idea_prompt = Prompts.prompt_task+Prompts.prompt_existing_idea.format(old_idea) + \
                              Prompts.prompt_topic.format(selected_topics)+Prompts.prompt_reference.format(paper_reference) + \
                              self.format_memories(current_memories, previous_memories, None) + \
                              Prompts.prompt_response

                format_idea_prompt = BaseMessage.make_user_message(role_name="user", content=idea_prompt)
                reply = await self._safe_step(agent,
                                              format_idea_prompt,
                                              timeout=6000.0,
                                              context="generate_idea")
                if reply is None:
                    print('No Response_idea')
                    continue
                current_memories.append(reply)
                # self.log_dialogue('user', idea_prompt)
                self.log_dialogue(agent.role_name, reply.content)
                old_idea = extract_between_json_tags(reply.content, num=1)
                idea_message = BaseMessage.make_user_message(role_name="user", content="In team {}, you come up with the idea: {}".format(self.team_name, old_idea))
                await agent.update_memory(idea_message, OpenAIBackendRole.USER)
                try:
                    if "Title" in old_idea:
                        idea_key = old_idea.split("Title")[1]
                        idea_key = strip_non_letters(idea_key.split("Experiment")[0])
                    else:
                        idea_key = old_idea.split("Idea")[1]
                        idea_key = strip_non_letters(idea_key.split("Experiment")[0])
                except:
                    idea_key = old_idea[:100]

                if len(idea_key)>=3:
                    try:
                        idea_key_prompt = BaseMessage.make_user_message(role_name="user", content=idea_key)
                        resp = await self._safe_embed(
                            teammate[0],
                            idea_key_prompt,
                            timeout=1500.0,
                            context="generate_idea_key",
                        )
                        if resp is None or getattr(resp, "data", None) is None:
                            raise ValueError("Embedding failed for idea_key.")
                        query_vector = np.array([resp.data[0].embedding])
                    except:
                        idea_key = "No topics."
                        idea_key_prompt = BaseMessage.make_user_message(role_name="user", content=idea_key)
                        resp = await self._safe_embed(
                            teammate[0],
                            idea_key_prompt,
                            timeout=1500.0,
                            context="generate_idea_key_fallback",
                        )
                        if resp is None or getattr(resp, "data", None) is None:
                            query_vector = np.random.rand(1, 1024)
                        else:
                            query_vector = np.array([resp.data[0].embedding])
                    paper_reference, cite_paper_new = await platform.reference_paper_alignment(query_vector, platform.cite_number, self.epoch)

                else:
                    paper_reference=''
                    cite_paper_new=[]
                cite_paper = list(set(cite_paper).union(cite_paper_new))

                # find the metric
                split_keywords = ['Clarity', 'Feasibility', 'Novelty']
                metrics = extract_metrics(old_idea, split_keywords)
                if best_idea != None:
                    if old_idea == best_idea:
                        idea_judge=True
                        # print("exit early!!!!!!")
                        break
                    best_metrics = extract_metrics(best_idea, split_keywords)
                    old_count = 0
                    best_count = 0
                    for split_keyword in split_keywords:
                        if metrics[split_keyword]==None:
                            break
                        if split_keyword=='Novelty':
                            old_count = old_count + 2*metrics[split_keyword]
                        else:
                            old_count = old_count + metrics[split_keyword]
                        if best_metrics[split_keyword]==None:
                            break
                        best_count = best_count + best_metrics[split_keyword]
                    if old_count>=best_count:
                        best_idea = old_idea
                        idea_list.append(old_idea)
                        mark_list.append(old_count)
                else:
                    idea_list.append(old_idea)
                    best_idea = old_idea
                # if all metrics are larger than 8, then over
                for split_keyword in split_keywords:
                    if metrics[split_keyword]==None:
                        break
                    if metrics[split_keyword]<10:
                        idea_judge=False
                        break
                if idea_judge:
                    best_idea=old_idea
                    break
            if idea_judge:
                break
            
            # summarize this turn's discussion
            turn_summarization_prompt = 'Briefly summarize "Discussion in this turn".' +\
                                        self.format_memories(current_memories, previous_memories, None)
            format_turn_summarization_prompt = BaseMessage.make_user_message(role_name="user", content=turn_summarization_prompt)

            random_summarizer_idea = random.choice(teammate)
            x_msg = await self._safe_step(random_summarizer_idea,
                                          format_turn_summarization_prompt,
                                          timeout=6000.0,
                                          context="generate_idea_summary")
            if x_msg is not None:
                self.log_dialogue(random_summarizer_idea.role_name, x_msg.content)
                turn_summarization = BaseMessage.make_user_message(
                    role_name="Summarization of turn{}".format(turn+1),
                    content=x_msg.content)
                previous_memories.append(turn_summarization)
                
        if self.idea == "None":
            if len(idea_list)>3:
                indices = top_three_indices(mark_list)
                idea_list = [idea_list[i] for i in indices]
                self.idea = idea_list
            else:
                self.idea = idea_list
        if len(self.idea)==0:
            self.idea = ['best_idea']
        # print("Candidate Idea:")
        # print(self.idea)
        if platform.skip_check:
            self.state=5
        else:
            self.state=4
        self.citation_id = cite_paper
        # print(len(self.citation_id))

    async def check_novelty(self, platform):
        teammate = platform.id_to_agent(self.teammate)
        existing_idea = self.idea
        idea_choices = ""
        for idea_index in range(len(existing_idea)):
            idea = existing_idea[idea_index]
            idea_choices = idea_choices+"Idea "+str(idea_index)+":\n"+idea+"\n"
        related_papers = []
        for idea_index in existing_idea:
            try:
                title = idea_index.split("Title")[1]
                title = strip_non_letters(title.split("Experiment")[0])
            except:
                title = idea_index[:100]
            if len(existing_idea)==3:
                cite_number = 3
            else:
                cite_number = 5
            if len(title)<5:
                related_paper=[]
            else:
                try:
                    title_prompt = BaseMessage.make_user_message(role_name="user", content=title)
                    resp = await self._safe_embed(
                        teammate[0],
                        title_prompt,
                        timeout=1500.0,
                        context="check_novelty_title",
                    )
                    if resp is None or getattr(resp, "data", None) is None:
                        raise ValueError("Embedding failed for title.")
                    query_vector = np.array([resp.data[0].embedding])
                except:
                    title = "No titles."
                    title_prompt = BaseMessage.make_user_message(role_name="user", content=title)
                    resp = await self._safe_embed(
                        teammate[0],
                        title_prompt,
                        timeout=1500.0,
                        context="check_novelty_title_fallback",
                    )
                    if resp is None or getattr(resp, "data", None) is None:
                        query_vector = np.random.rand(1, 1024)
                    else:
                        query_vector = np.array([resp.data[0].embedding])
                _, related_paper = await platform.reference_paper(query_vector, cite_number, self.epoch)

            related_papers = list(set(related_papers).union(related_paper))

        paper_reference = ""
        for id in range(len(related_papers)):
            paper_index = related_papers[id]
            paper_reference = paper_reference+"Paper {}:".format(id+1)+"\n"
            paper_reference = paper_reference+"Title: "+platform.paper_dicts[paper_index]['title']+"\n"
            paper_reference = paper_reference+"Abstract: "+platform.paper_dicts[paper_index]['abstract']+"}"+"\n"
        
        idea_novelty_prompt = Prompts.prompt_idea_check + \
                                Prompts.prompt_idea_check_response.replace("{existing_idea}", idea_choices).replace("{last_query_results}", paper_reference)
        format_idea_novelty_prompt = BaseMessage.make_user_message(role_name="user", content=idea_novelty_prompt)
        random_agent_id_3 = random.choice(self.teammate)
        reply_msg = await self._safe_step(
            platform.id2agent[random_agent_id_3],
            format_idea_novelty_prompt,
            timeout=6000.0,
            context="check_novelty",
        )
        if reply_msg is None:
            reply_text = "```No Response_novelty```"
        else:
            reply_text = reply_msg.content
        self.log_dialogue(platform.id2agent[random_agent_id_3].role_name, reply_text)
        old_idea = extract_between_json_tags(reply_text, num=1)
        idea_choice = extract_first_number(old_idea)

        try:
            self.idea = existing_idea[idea_choice]
        except:
            self.idea = existing_idea[0]
        if self.idea == "None":
            self.idea = 'No idea and you can think freely.'
        if len(self.idea)<10:
            self.idea = 'No idea and you can think freely.'

        # print("Final Idea:")
        # print(self.idea)
        self.state=5

    async def generate_abstract(self, platform):
        idea = self.idea
        old_abstract = self.abstract
        teammate = platform.id_to_agent(self.teammate)

        if len(teammate)==1:
            group_max_discuss_iteration = platform.group_max_discuss_iteration
        else:
            group_max_discuss_iteration = platform.group_max_discuss_iteration

        for turn in range(group_max_discuss_iteration):
            # discuss the abstract
            for agent_id in range(len(teammate)):
                if old_abstract == "None":
                    abstract_prompt = Prompts.prompt_abstract+"\n"+\
                                      idea+"\n"+\
                                      Prompts.prompt_abstract_requirement+"\n"+\
                                      Prompts.prompt_abstract_response
                else:
                    # the paper is not reviewed by reviewer
                    if self.paper_review == "None":
                        # the paper is not reviewer by the team member
                        if self.self_review == "None":
                            prompt_abstract_judgement = Prompts.prompt_abstract_judgement.replace("[Insert abstract here]",old_abstract)
                            abstract_prompt = prompt_abstract_judgement+Prompts.prompt_abstract_revise_response
                        else:
                            prompt_abstract_judgement = Prompts.prompt_abstract_judgement_self.replace("[Insert abstract here]",old_abstract)
                            prompt_abstract_judgement = prompt_abstract_judgement.replace("[Insert self_review comments]", self.self_review)
                            abstract_prompt = prompt_abstract_judgement+Prompts.prompt_abstract_revise_response
                    else:
                        prompt_abstract_judgement = Prompts.prompt_abstract_judgement_after_review.replace("[Insert Reviewer comments]",self.paper_review)
                        prompt_abstract_judgement = prompt_abstract_judgement.replace("[Insert abstract here]",old_abstract)
                        abstract_prompt = prompt_abstract_judgement+Prompts.prompt_abstract_revise_response

                format_abstract_prompt = BaseMessage.make_user_message(role_name="user", content=abstract_prompt)
                reply = await self._safe_step(teammate[agent_id],
                                              format_abstract_prompt,
                                              timeout=6000.0,
                                              context="generate_abstract")
                if reply is None:
                    continue
                    # print(f'---------teamname{self.team_name}-------{format_abstract_prompt}')
                    # print(f'---------teamname{self.team_name}-------No Response_abstract')
                self.log_dialogue(teammate[agent_id].role_name, reply.content)
                old_old_abstract = old_abstract
                old_abstract = extract_between_json_tags(reply.content, num=1)
                abstract_message = BaseMessage.make_user_message(role_name="user", content="In team {}, you come up with the abstract: {}".format(self.team_name, old_abstract))
                await teammate[agent_id].update_memory(abstract_message, OpenAIBackendRole.USER)
                if len(old_abstract.split("Abstract"))<2:
                    if agent_id!=0:
                        old_abstract=old_old_abstract
                    else:
                        old_abstract = reply.content

        related_papers = []

        try:
            Abstract = strip_non_letters(old_abstract.split("Abstract")[1])
        except:
            Abstract = "No abstracts."

        try:
            abstract_prompt = BaseMessage.make_user_message(role_name="user", content=Abstract)
            resp = await self._safe_embed(
                teammate[0],
                abstract_prompt,
                timeout=1500.0,
                context="generate_abstract_query",
            )
            if resp is None or getattr(resp, "data", None) is None:
                raise ValueError("Embedding failed for abstract.")
            query_vector = np.array([resp.data[0].embedding])
        except:
            Abstract = "No abstracts."
            abstract_prompt = BaseMessage.make_user_message(role_name="user", content=Abstract)
            resp = await self._safe_embed(
                teammate[0],
                abstract_prompt,
                timeout=1500.0,
                context="generate_abstract_query_fallback",
            )
            if resp is None or getattr(resp, "data", None) is None:
                query_vector = np.random.rand(1, 1024)
            else:
                query_vector = np.array([resp.data[0].embedding])
        cite_number_temp=8
        D_future, I_future = platform.gpu_future_index.search(query_vector, int(cite_number_temp/2))
        D, I = platform.gpu_index.search(query_vector, int(cite_number_temp/2))

        for id in range(len(I_future[0])):
            paper_title = platform.paper_future_dicts[I_future[0][id]]['title']
            paper_abstract = platform.paper_future_dicts[I_future[0][id]]['abstract']
            paper_year = platform.paper_future_dicts[I_future[0][id]]['year']
            paper_citation = platform.paper_future_dicts[I_future[0][id]]['citation']
            paper_index = {}
            paper_index['title'] = paper_title
            paper_index['abstract'] = paper_abstract
            paper_index['year'] = paper_year
            paper_index['citation'] = paper_citation
            related_papers.append(paper_index)

        for id in range(len(I[0])):
            paper_title = platform.paper_dicts[I[0][id]]['title']
            paper_abstract = platform.paper_dicts[I[0][id]]['abstract']
            paper_year = platform.paper_dicts[I[0][id]]['year']
            paper_citation = platform.paper_dicts[I[0][id]]['citation']
            paper_index = {}
            paper_index['title'] = paper_title
            paper_index['abstract'] = paper_abstract
            paper_index['year'] = paper_year
            paper_index['citation'] = paper_citation
            related_papers.append(paper_index)

        # eval with embedding similarity
        abs = []
        our_abs = strip_non_letters(old_abstract.split('Abstract')[1])

        try:
            our_abs_prompt = BaseMessage.make_user_message(role_name="user", content=our_abs)
            resp = await self._safe_embed(
                teammate[0],
                our_abs_prompt,
                timeout=1500.0,
                context="our_abstract_embed",
            )
            if resp is None or getattr(resp, "data", None) is None:
                raise ValueError("Embedding failed for our_abs.")
            query_vector = resp.data[0].embedding
        except:
            our_abs = "No abstracts."
            our_abs_prompt = BaseMessage.make_user_message(role_name="user", content=our_abs)
            resp = await self._safe_embed(
                teammate[0],
                our_abs_prompt,
                timeout=1500.0,
                context="our_abstract_embed_fallback",
            )
            if resp is None or getattr(resp, "data", None) is None:
                query_vector = np.random.rand(1024)
            else:
                query_vector = resp.data[0].embedding

        abs.append(query_vector)

        for paper_id in range(len(related_papers)):
            related_astract = related_papers[paper_id]['abstract']
            try:
                related_astract_prompt = BaseMessage.make_user_message(role_name="user", content=related_astract)
                resp = await self._safe_embed(
                    teammate[0],
                    related_astract_prompt,
                    timeout=1500.0,
                    context="related_abstract_embed",
                )
                if resp is None or getattr(resp, "data", None) is None:
                    raise ValueError("Embedding failed for related_abstract.")
                query_vector = resp.data[0].embedding
            except:
                related_astract = "No abstracts."
                related_astract_prompt = BaseMessage.make_user_message(role_name="user", content=related_astract)
                resp = await self._safe_embed(
                    teammate[0],
                    related_astract_prompt,
                    timeout=1500.0,
                    context="related_abstract_embed_fallback",
                )
                if resp is None or getattr(resp, "data", None) is None:
                    query_vector = np.random.rand(1024)
                else:
                    query_vector = resp.data[0].embedding
            abs.append(query_vector)

        sim = []
        for emb_id in range(1, len(abs)):
            sim.append(torch.nn.functional.cosine_similarity(torch.tensor(abs[0]).unsqueeze(0),
                                                             torch.tensor(abs[emb_id]).unsqueeze(0), dim=-1)[0].item())
        self.log_dialogue('embedding similarity', str(sim))

        self.log_dialogue('faiss_distance', str(D))
        self.log_dialogue('faiss_distance_future', str(D_future))

        # eval with LLM
        # print('related papers:')
        # print(len(related_papers))
        if len(related_papers)>0:
            self.log_dialogue('arxiv',related_papers)
        # find paper successfully
        if len(related_papers)>0:
            abstract_check_prompt = Prompts.prompt_abstract_check.replace("[Insert your abstract here]", old_abstract)
            cite_abstract = ""
            word = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
            split_keywords = []
            for paper_id in range(int(cite_number_temp/2), len(related_papers)):
                cite_abstract = cite_abstract + str(paper_id-int(cite_number_temp/2)+1) + ". Abstract {}: ".format(word[paper_id-int(cite_number_temp/2)]) + "Title: " + related_papers[paper_id]['title'] + "\n" + "Abstract: " + related_papers[paper_id]['abstract'] + "\n"
                split_keywords.append('Written Abstract vs {}'.format(word[paper_id-int(cite_number_temp/2)]))
            abstract_check_prompt = abstract_check_prompt.replace("[Insert ref abstract here]", cite_abstract)
            abstract_check_prompt = abstract_check_prompt + "\n" + Prompts.prompt_response_check

            format_abstract_check_prompt = BaseMessage.make_user_message(role_name="user", content=abstract_check_prompt)
            leader_index = random.randint(0, len(self.teammate)-1)
            reply = await self._safe_step(teammate[leader_index],
                                          format_abstract_check_prompt,
                                          timeout=6000.0,
                                          context="abstract_check")
            if reply is None:
                self.abstract = old_abstract
                self.state = 6
                return
            self.log_dialogue(teammate[leader_index].role_name, reply.content)
            # print("abstract_check:")
            # print(split_keywords)
            comparison = extract_between_json_tags(reply.content)
            metric = extract_metrics(comparison, split_keywords=split_keywords)
            abstract_use = True
            try:
                for split_keyword in split_keywords:
                    if metric[split_keyword]>=96:
                        abstract_use = False
                        self.abstract = old_abstract
                        break
            except:
                abstract_use = False
            self.abstract = old_abstract
            # print('Final Abstract:')
            # print(self.abstract)
            # stop early
            # self.state=7

            # do not stop early

            if abstract_use:
                self.state=6
                self.self_review="None"
            # if the abstract is too similar one time, go to revise, otherwise back to generate idea
            else:
                if self.self_review!="None":
                    self.state=3
                    self.idea = "None"
                    self.abstract = "None"
                    self.citation_id = []
                    self.self_review = "None"
                    self.paper_review = "None"
                else:
                    self.self_review = reply.content

        else:
            # print('Check Fail!!!!!!')
            if self.abstract == "None":
                self.abstract = old_abstract
                # print('Final Abstract:')
                # print(self.abstract)
                self.state=6

    async def generate_review(self, platform):
        teammate = platform.id_to_agent(self.teammate)
        # paper reviewer by reviewer
        # print('current reviewing paper from {}'.format(self.teammate))
        old_abstract = self.abstract
        review_prompt = Prompts.prompt_review_require_simple.replace("{paper}", old_abstract)
        mark_sum = 0
        self.paper_review = "None"
        for _ in range(platform.reviewer_num):
            format_review_prompt = BaseMessage.make_user_message(role_name="user", content=review_prompt)
            reply = await self._safe_step(platform.reviewer_pool[_],
                                          format_review_prompt,
                                          timeout=6000.0,
                                          context="paper_review")
            if reply is None:
                self.log_dialogue(platform.reviewer_pool[_].role_name,
                                  "(No response_review, use default mark)")
                split_keywords = ['Overall']
                metric = {'Overall': None}
            else:
                self.log_dialogue(platform.reviewer_pool[_].role_name, reply.content)
                split_keywords = ['Overall']
                metric = extract_metrics(reply.content, split_keywords)
            if self.paper_review == "None":
                self.paper_review = platform.reviewer_pool[_].role_name+":\n"+reply.content
            else:
                self.paper_review = self.paper_review+"\n"+platform.reviewer_pool[_].role_name+":\n"+reply.content
            for split_keyword in split_keywords:
                if metric[split_keyword] is None:
                    mark_sum = mark_sum + platform.default_mark
                else:
                    mark_sum = mark_sum + metric[split_keyword]
        self.mark_history.append(mark_sum)
        if mark_sum>=(6*platform.reviewer_num):
            # print('paper accept!!!!!!')
            self.state=platform.over_state
            try:
                title = old_abstract.split("Abstract")[0]
                title = strip_non_letters(title.split("Title")[1])
                abstract = strip_non_letters(old_abstract.split("Abstract")[1])
            except:
                title = "No titles."
                abstract = "No abstracts."
            # add discipline
            disciplines = ['art', 'biology', 'business', 'computer science', 'chemistry', 'economics', 'engineering', 'environmental science',
                'geography', 'geology', 'history', 'materials science', 'mathematics', 'medicine', 'philosophy', 'physics', 'political science',
                'psychology', 'sociology']
            try:
                discipline_prompt = Prompts.prompt_discipline.replace('[ABSTRACT]', abstract)
                format_discipline_prompt = BaseMessage.make_user_message(role_name="user", content=discipline_prompt)
                random_agent = random.choice(teammate)
                reply = await random_agent.step(format_discipline_prompt)
                reply = reply.msg.content.lower()
                discipline = find_best_match(filter_out_number_n_symbol(reply), disciplines)
            except:
                discipline = 'computer science'
            try:
                keywords_prompt = Prompts.prompt_keywords.replace('[ABSTRACT]', abstract)
                format_keyword_prompt = BaseMessage.make_user_message(role_name="user", content=keywords_prompt)
                random_agent = random.choice(teammate)
                reply = await random_agent.step(format_keyword_prompt)
                keywords = extract_between_json_tags(reply.msg.content)
            except:
                keywords = 'None'
            file_dict={}
            file_dict['title']=title
            file_dict['abstract']=abstract
            file_dict['year']=self.epoch
            file_dict['citation']=0
            file_dict['id'] = len(platform.paper_dicts)
            file_dict['authors'] = self.teammate
            file_dict['cite_papers'] = self.citation_id
            file_dict['reviews'] = self.mark_history
            file_dict['discipline'] = discipline
            file_dict['keywords'] = keywords
            platform.paper_dicts.append(file_dict)
            platform.paper_citation_list[file_dict['id']] = 0
            # add embedding into list
            embedding_list = []

            try:
                abstract_prompt = BaseMessage.make_user_message(role_name="user", content=abstract)
                resp = await self._safe_embed(
                    platform.reviewer_pool[0],
                    abstract_prompt,
                    timeout=1500.0,
                    context="reviewer_abstract_embed",
                )
                if resp is None or getattr(resp, "data", None) is None:
                    raise ValueError("Embedding failed for reviewer abstract.")
                query_vector = resp.data[0].embedding
            except:
                abstract = "No abstracts."
                abstract_prompt = BaseMessage.make_user_message(role_name="user", content=abstract)
                resp = await self._safe_embed(
                    platform.reviewer_pool[0],
                    abstract_prompt,
                    timeout=1500.0,
                    context="reviewer_abstract_embed_fallback",
                )
                if resp is None or getattr(resp, "data", None) is None:
                    query_vector = np.random.rand(1024)
                else:
                    query_vector = resp.data[0].embedding

            embedding_list.append(query_vector)
            response = np.array(embedding_list)
            platform.gpu_index.add(response)
        else:
            if len(self.mark_history)>=2:
                failure_check_prompt = Prompts.prompt_failure_check.replace("[insert failure times]", str(len(self.mark_history)))
                failure_check_prompt = failure_check_prompt.replace("[insert failure reviews]", str(self.paper_review))
                failure_check_prompt = failure_check_prompt.replace("[insert each failure]", str(self.mark_history))
                format_failure_check_prompt = BaseMessage.make_user_message(role_name="user", content=failure_check_prompt)
                random_agent = random.choice(teammate)
                reply = await random_agent.step(format_failure_check_prompt)
                reply = reply.msg
                self.log_dialogue(random_agent.role_name, reply.content)
                answer_pattern = re.compile(r'2', re.IGNORECASE)

                # check whether agent is ready to answer
                if answer_pattern.search(reply.content):
                    self.state=platform.over_state
                    try:
                        title = old_abstract.split("Abstract")[0]
                        title = strip_non_letters(title.split("Title")[1])
                        abstract = strip_non_letters(old_abstract.split("Abstract")[1])
                    except:
                        title = "No titles."
                        abstract = "No abstracts."
                    # add discipline
                    disciplines = ['art', 'biology', 'business', 'computer science', 'chemistry', 'economics', 'engineering', 'environmental science',
                        'geography', 'geology', 'history', 'materials science', 'mathematics', 'medicine', 'philosophy', 'physics', 'political science',
                        'psychology', 'sociology']
                    try:
                        discipline_prompt = Prompts.prompt_discipline.replace('[ABSTRACT]', abstract)
                        format_discipline_prompt = BaseMessage.make_user_message(role_name="user", content=discipline_prompt)
                        random_agent = random.choice(teammate)
                        reply = await random_agent.step(format_discipline_prompt)
                        reply = reply.msg.content.lower()
                        discipline = find_best_match(filter_out_number_n_symbol(reply), disciplines)
                    except:
                        discipline = 'computer science'
                    try:
                        keywords_prompt = Prompts.prompt_keywords.replace('[ABSTRACT]', abstract)
                        format_keyword_prompt = BaseMessage.make_user_message(role_name="user", content=keywords_prompt)
                        random_agent = random.choice(teammate)
                        reply = await random_agent.step(format_keyword_prompt)
                        keywords = extract_between_json_tags(reply.msg.content)
                    except:
                        keywords = 'None'
                    file_dict={}
                    file_dict['title']=title
                    file_dict['abstract']=abstract
                    file_dict['year']=self.epoch
                    file_dict['citation']=0
                    file_dict['id'] = len(platform.paper_dicts)
                    file_dict['authors'] = self.teammate
                    file_dict['cite_papers'] = self.citation_id
                    file_dict['reviews'] = self.mark_history
                    file_dict['discipline'] = discipline
                    file_dict['keywords'] = keywords
                    platform.paper_dicts.append(file_dict)
                    platform.paper_citation_list[file_dict['id']] = 0
                    # add embedding into list
                    embedding_list = []

                    try:
                        abstract_prompt = BaseMessage.make_user_message(role_name="user", content=abstract)
                        resp = await self._safe_embed(
                            platform.reviewer_pool[0],
                            abstract_prompt,
                            timeout=1500.0,
                            context="reviewer_abstract_embed_failure",
                        )
                        if resp is None or getattr(resp, "data", None) is None:
                            raise ValueError("Embedding failed for reviewer abstract in failure branch.")
                        query_vector = resp.data[0].embedding
                    except:
                        abstract = "No abstracts."
                        abstract_prompt = BaseMessage.make_user_message(role_name="user", content=abstract)
                        resp = await self._safe_embed(
                            platform.reviewer_pool[0],
                            abstract_prompt,
                            timeout=1500.0,
                            context="reviewer_abstract_embed_failure_fallback",
                        )
                        if resp is None or getattr(resp, "data", None) is None:
                            query_vector = np.random.rand(1024)
                        else:
                            query_vector = resp.data[0].embedding

                    embedding_list.append(query_vector)
                    response = np.array(embedding_list)
                    platform.gpu_index.add(response)
                else:
                    self.state = 5
            else:
                self.state = 5

    def log_dialogue(self, name, content):
        color = Color.GREEN
        # name_after = f"{color}{name}{Color.RESET}"
        # print(f'{name_after}: {content}')
        # 全局
        # self.logger_2.info(f'{name}: {content}')
        # print(f'-'*30)
        # self.logger.info(f'{"="*50} Epoch:{self.epoch} | {self.state_log[self.state]} | {name} {"="*50}\n{content}')
        # self.logger.info(f'{"="*100}')

    def save_team_info(self):
        team_info = {
            'teammate':self.teammate,
            'topic':self.topic,
            'idea':self.idea,
            'abstract':self.abstract
        }
        # print(f'{"="*50} SAVE TEAM INFO {"="*50}')
        # with open(self.info_file, 'w') as json_file:
        #     json.dump(team_info, json_file, indent=4)
    
    def save_to_file(self,path):
        with open(path, "a", encoding="utf-8") as f:
            team_dict={
            'team_name':self.team_name,
            'state':str(self.state),
            'epoch':str(self.epoch),
            'teammate':self.teammate,
            'mark_history':[str(mark) for mark in self.mark_history],
            'recent_n_team_mem_for_retrieve':str(self.recent_n_team_mem_for_retrieve),
            'topic':self.topic,
            'idea':self.idea,
            'abstract':self.abstract,
            'citation_id':[str(id) for id in self.citation_id],
            'self_review':self.self_review,
            'paper_review':self.paper_review,
            'log_dir':self.log_dir,
            'info_dir':self.info_dir
            }
            f.write(json.dumps(team_dict)+"\n")
    
    @classmethod
    def load_from_file(cls,team_dict):
        return cls(team_dict['team_name'],
                   team_dict['log_dir'],
                   team_dict['info_dir'],
                   int(team_dict['recent_n_team_mem_for_retrieve']),
                   int(team_dict['state']),
                   int(team_dict['epoch']),
                   team_dict['teammate'],
                   [], #memory
                   [int(mark) for mark in team_dict['mark_history']],
                   team_dict['topic'],
                   team_dict['idea'],
                   team_dict['abstract'],
                   [int(id) for id in team_dict['citation_id']],
                   team_dict['self_review'],
                   team_dict['paper_review'])

if __name__=='__main__':
    team1 = Team('LPL')
    team2 = Team('LCK')
    team1.log_dialogue('sam', 'LPL win!')
    team2.log_dialogue('tom', 'LCK win!')
    team1.log_dialogue('sam', 'LPL win again !')
    team2.log_dialogue('tom', 'LCK win again !')