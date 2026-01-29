from datetime import datetime
from copy import deepcopy
import logging
import re
import ollama
import torch.nn.functional
import numpy as np
import json
import os
import sys
import random
import heapq
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
        teammate = platform.id_to_agent(self.teammate)

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
                
                agent_prompt = f"Current team members are {','.join(self.teammate)}.\n" + \
                            self.format_memories(None, None, team_memories) + \
                            prompt + \
                            self.format_memories(current_memories, previous_memories, None)
                format_agent_prompt = BaseMessage.make_user_message(role_name="user", content=agent_prompt)

                # add reply to turn_history
                try:
                    reply = await agent.step(format_agent_prompt)
                    reply = reply.msg
                except:
                    reply = None
                    print('No response_discussion')
                if reply is None:
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
                print(turn_summarization)
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
        answer = await platform.id2agent[self.teammate[random_agent_id]].step(format_answer_prompt)
        answer = answer.msg
        # self.log_dialogue('user', answer_prompt)
        self.log_dialogue(platform.id2agent[self.teammate[random_agent_id]].role_name, answer.content)
        answer_pattern = re.compile(r'1', re.IGNORECASE)

        # check whether agent is ready to answer
        if answer_pattern.search(answer.content) or len(team_memories)>=1:
            self.state = 3
            # prompt
            topic_prompt = Prompts.to_ask_topic.replace("[history_prompt]", self.format_memories(current_memories, previous_memories, team_memories))
            format_topic_prompt = BaseMessage.make_user_message(role_name="user", content=topic_prompt)
            # answer
            leader_index=random.randint(0, len(self.teammate)-1)
            topic = await platform.id2agent[self.teammate[leader_index]].step(format_topic_prompt)
            topic = topic.msg
            self.log_dialogue(platform.id2agent[self.teammate[leader_index]].role_name, topic.content)
            try:
                self.topic = extract_between_json_tags(topic.content, num=1)
                self.topic = strip_non_letters(self.topic.split("Selected Topic")[1])
                if len(self.topic)<3:
                    self.topic = topic.content
                    self.topic = strip_non_letters(self.topic.split("Selected Topic")[1])
            except:
                self.topic = topic.content
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
        dialogue_summarization = await platform.id2agent[random_agent_id_2].step(format_dialogue_summarization_prompt)
        dialogue_summarization = dialogue_summarization.msg
        team_memories.append(dialogue_summarization)
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

        for turn in range(group_max_discuss_iteration):
            current_memories = []
            # discuss the idea
            for agent in teammate:
                idea_prompt = Prompts.prompt_task+Prompts.prompt_existing_idea.format(old_idea) + \
                              Prompts.prompt_topic.format(selected_topics)+ \
                              self.format_memories(current_memories, previous_memories, None) + \
                              Prompts.prompt_response

                format_idea_prompt = BaseMessage.make_user_message(role_name="user", content=idea_prompt)
                try:
                    reply = await agent.step(format_idea_prompt)
                    reply = reply.msg
                except:
                    reply = None
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

                # find the metric
                split_keywords = ['Clarity', 'Feasibility', 'Novelty']
                metrics = extract_metrics(old_idea, split_keywords)
                if best_idea != None:
                    if old_idea == best_idea:
                        idea_judge=True
                        print("exit early!!!!!!")
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
            x = await random_summarizer_idea.step(format_turn_summarization_prompt)
            x = x.msg
            self.log_dialogue(random_summarizer_idea.role_name, x.content)
            turn_summarization = BaseMessage.make_user_message(role_name="Summarization of turn{}".format(turn+1), content=x.content)

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
        print("Candidate Idea:")
        print(self.idea)
        if platform.skip_check:
            self.state=5
        else:
            self.state=4

    async def check_novelty(self, platform):
        teammate = platform.id_to_agent(self.teammate)
        existing_idea = self.idea
        idea_choices = ""
        for idea_index in range(len(existing_idea)):
            idea = existing_idea[idea_index]
            idea_choices = idea_choices+"Idea "+str(idea_index)+":\n"+idea+"\n"
        

        idea_novelty_prompt = Prompts.prompt_idea_check_no_reference + \
                                      Prompts.prompt_idea_check_response_no_reference.replace("{existing_idea}", idea_choices)
        format_idea_novelty_prompt = BaseMessage.make_user_message(role_name="user", content=idea_novelty_prompt)
        random_agent_id_3 = random.choice(self.teammate)
        try:
            reply = await platform.id2agent[random_agent_id_3].step(format_idea_novelty_prompt)
            reply = reply.msg.content
        except:
            reply = "```No Response_novelty```"
        self.log_dialogue(platform.id2agent[random_agent_id_3].role_name, reply)
        old_idea = extract_between_json_tags(reply, num=1)
        idea_choice = extract_first_number(old_idea)

        try:
            self.idea = existing_idea[idea_choice]
        except:
            self.idea = existing_idea[0]
        if self.idea == "None":
            self.idea = 'No idea and you can think freely.'
        if len(self.idea)<10:
            self.idea = 'No idea and you can think freely.'

        print("Final Idea:")
        print(self.idea)
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
                    if self.paper_review == "None":
                        if self.self_review == "None":
                            prompt_abstract_judgement = Prompts.prompt_abstract_judgement.replace("[Insert abstract here]",old_abstract)
                            abstract_prompt = prompt_abstract_judgement+Prompts.prompt_abstract_revise_response
                        else:
                            prompt_abstract_judgement = Prompts.prompt_abstract_judgement_self.replace("[Insert abstract here]",old_abstract)
                            prompt_abstract_judgement = prompt_abstract_judgement.replace("[Insert self_review comments]", self.self_review)
                            abstract_prompt = prompt_abstract_judgement+Prompts.prompt_abstract_revise_response

                format_abstract_prompt = BaseMessage.make_user_message(role_name="user", content=abstract_prompt)
                try:
                    reply = await teammate[agent_id].step(format_abstract_prompt)
                    reply = reply.msg
                except:
                    reply = None
                    continue

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

        if self.abstract == "None":
            self.abstract = old_abstract
            print('Final Abstract:')
            print(self.abstract)
            self.state=6

    async def generate_review(self, platform):
        teammate = platform.id_to_agent(self.teammate)
        old_abstract = self.abstract
        
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
        random_agent_id_4 = random.choice(self.teammate)
        try:
            discipline_prompt = Prompts.prompt_discipline.replace('[ABSTRACT]', abstract)
            format_discipline_prompt = BaseMessage.make_user_message(role_name="user", content=discipline_prompt)
            reply = await platform.id2agent[random_agent_id_4].step(format_discipline_prompt)
            reply = reply.msg.content.lower()
            discipline = find_best_match(filter_out_number_n_symbol(reply), disciplines)
        except:
            discipline = 'computer science'
        try:
            keywords_prompt = Prompts.prompt_keywords.replace('[ABSTRACT]', abstract)
            format_keyword_prompt = BaseMessage.make_user_message(role_name="user", content=keywords_prompt)
            reply = await platform.id2agent[random_agent_id_4].step(format_keyword_prompt)
            keywords = extract_between_json_tags(reply.msg.content)
        except:
            keywords = 'None'
        file_dict={}
        file_dict['title']=title
        file_dict['abstract']=abstract
        file_dict['year']=self.epoch
        file_dict['id'] = len(platform.paper_dicts)
        file_dict['authors'] = self.teammate
        file_dict['discipline'] = discipline
        file_dict['keywords'] = keywords
        platform.paper_dicts.append(file_dict)

    def log_dialogue(self, name, content):
        color = Color.GREEN
        name_after = f"{color}{name}{Color.RESET}"
        print(f'{name_after}: {content}')
        # 全局
        # self.logger_2.info(f'{name}: {content}')
        print(f'-'*30)
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
            'recent_n_team_mem_for_retrieve':str(self.recent_n_team_mem_for_retrieve),
            'topic':self.topic,
            'idea':self.idea,
            'abstract':self.abstract,
            'self_review':self.self_review,
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
                   team_dict['topic'],
                   team_dict['idea'],
                   team_dict['abstract'],
                   team_dict['self_review'])

if __name__=='__main__':
    team1 = Team('LPL')
    team2 = Team('LCK')
    team1.log_dialogue('sam', 'LPL win!')
    team2.log_dialogue('tom', 'LCK win!')
    team1.log_dialogue('sam', 'LPL win again !')
    team2.log_dialogue('tom', 'LCK win again !')