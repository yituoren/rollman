import json
from utils.utils import write_to_judger

def create_message(operation: list) -> str:
    if len(operation) != 3:
        raise ValueError("operation 列表的长度必须为3")
    
    message = {
        "role": 1,
        "operation": operation
    }
    return json.dumps(message)

# put in operation num for each ghost (0: stay, 1: up, 2: left, 3: down, 4: right)
def send_message(op1: int, op2: int, op3: int):
    operation = [op1, op2, op3]
    msg = create_message(operation)
    write_to_judger(msg)