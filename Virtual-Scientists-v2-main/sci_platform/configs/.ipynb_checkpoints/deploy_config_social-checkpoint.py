# deploy setups
ips=['127.0.0.1']
port = list(range(11461, 11463))

# exp setups
agent_num = 1000
runs = 1
team_limit = 3
max_discuss_iteration = 2
max_team_member = 14
epochs = 30
# model_name = "/inspire/hdd/project/aiscientist/hufang-CZXS24220079/MODEL/Qwen3-235B-A22B-Instruct-2507-FP8"
model_name = "/inspire/ssd/project/aiscientist/public/MODEL/Llama-3.1-70B-Instruct"
# model_name = "/inspire/hdd/project/aiscientist/hufang-CZXS24220079/MODEL/Qwen3-30B-A3B-Instruct-2507-FP8"
# model_name = "/inspire/hdd/project/aiscientist/hufang-CZXS24220079/MODEL/gemma-3-27b-it"
# model_name = "/inspire/hdd/project/aiscientist/hufang-CZXS24220079/MODEL/DeepSeek-R1-Distill-Llama-70B"

leader_mode = 'normal' 

# checkpoint setups
checkpoint = False
test_time = 'Llama-3.1-70B-Instruct-social'
load_time = 'Llama-3.1-70B-Instruct-social'