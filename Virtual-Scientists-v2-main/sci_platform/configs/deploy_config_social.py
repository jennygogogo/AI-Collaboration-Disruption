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

This file is a new configuration file for social platform deployment settings.
"""

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
model_name = "MODEL/Qwen3-235B-A22B-Instruct-2507-FP8"

leader_mode = 'normal' 

# checkpoint setups
checkpoint = False
test_time = 'Qwen3-235B-A22B-Instruct-2507-FP8-social'
load_time = 'Qwen3-235B-A22B-Instruct-2507-FP8-social'