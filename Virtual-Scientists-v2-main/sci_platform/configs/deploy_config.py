# deploy setups
ips=['127.0.0.1']
port = list(range(11461, 11463))

# exp setups
agent_num = 1000
runs = 1
team_limit = 3
max_discuss_iteration = 2
max_team_member = 6
epochs = 30
model_name = "MODEL/Qwen3-235B-A22B-Instruct-2507-FP8"

leader_mode = 'normal' 

# checkpoint setups
checkpoint = False
test_time = 'Qwen3-235B-A22B-Instruct-2507-FP8-fast'
load_time = 'Qwen3-235B-A22B-Instruct-2507-FP8-fast'