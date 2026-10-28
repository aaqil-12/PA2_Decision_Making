
import builtins

trace_file = open("Part1_Trace.txt", "w", encoding="utf-8")

def print(*args, **kwargs):
    builtins.print(*args, **kwargs)
    builtins.print(*args, **kwargs, file=trace_file)


states = [
    'RU_8p',
    'TU_10p', 'RU_10p', 'RD_10p',
    'RU_8a', 'RD_8a',
    'TU_10a', 'RU_10a', 'RD_10a', 'TD_10a',
    'Terminal'
]

transitions = {
    'RU_8p': {
        'P': [(1.0, +2, 'TU_10p')],
        'R': [(1.0, 0, 'RU_10p')],
        'S': [(1.0, -1, 'RD_10p')]
    },
    'TU_10p': {
        'P': [(1.0, +2, 'RU_10a')],
        'R': [(1.0, 0, 'RU_8a')]
    },
    'RU_10p': {
        'R': [(1.0, 0, 'RU_8a')],
        'P': [(0.5, +2, 'RU_8a'), (0.5, +2, 'RU_10a')],
        'S': [(1.0, -1, 'RD_8a')]
    },
    'RD_10p': {
        'R': [(1.0, 0, 'RD_8a')],
        'P': [(0.5, +2, 'RD_8a'), (0.5, +2, 'RD_10a')]
    },
    'RU_8a': {
        'P': [(1.0, +2, 'TU_10a')],
        'R': [(1.0, 0, 'RU_10a')],
        'S': [(1.0, -1, 'RD_10a')]
    },
    'RD_8a': {
        'R': [(1.0, 0, 'RD_10a')],
        'P': [(1.0, +2, 'TD_10a')]
    },
    'TU_10a': {
        'any': [(1.0, -1, 'Terminal')]
    },
    'RU_10a': {
        'any': [(1.0, 0, 'Terminal')]
    },
    'RD_10a': {
        'any': [(1.0, +4, 'Terminal')]
    },
    'TD_10a': {
        'any': [(1.0, +3, 'Terminal')]
    }
}

gamma = 0.99
theta = 0.001

V = {s: 0.0 for s in states}

iteration = 0

while True:
    iteration += 1
    max_change = 0.0

    print(f"\nITERATION {iteration}")

    V_new = V.copy()

    for s in states:
        if s == 'Terminal':
            continue

        prev_val = V[s]
        action_values = {}

        for action, trans_list in transitions[s].items():
            q_val = 0.0

            for prob, reward, next_state in trans_list:
                q_val += prob * (reward + gamma * V[next_state])

            action_values[action] = q_val

        best_action = max(action_values, key=action_values.get)
        new_val = action_values[best_action]

        V_new[s] = new_val

        change = abs(new_val - prev_val)
        max_change = max(max_change, change)

        q_str = ", ".join(
            f"{act}: {val:.4f}"
            for act, val in action_values.items()
        )

        print(
            f"State: {s:<8} | "
            f"Previous Value: {prev_val:.4f} | "
            f"New Value: {new_val:.4f} | "
            f"Action Values: [{q_str}] | "
            f"Selected Action: {best_action}"
        )

    V = V_new

    print(f"\nMaximum Change: {max_change:.6f}")

    if max_change < theta:
        break


print("FINAL RESULTS")


print(f"\nTotal Iterations: {iteration}")

print("\nFinal State Values:")

for s in states:
    print(f"{s:<10} : {V[s]:.4f}")

print("\nFinal Optimal Policy:")

for s in states:
    if s == 'Terminal':
        print(f"{s:<10} -> No Action")
        continue

    action_values = {}

    for action, trans_list in transitions[s].items():
        q_val = sum(
            prob * (reward + gamma * V[next_state])
            for prob, reward, next_state in trans_list
        )

        action_values[action] = q_val

    best_action = max(action_values, key=action_values.get)

    print(f"{s:<10} -> {best_action}")

trace_file.close()
