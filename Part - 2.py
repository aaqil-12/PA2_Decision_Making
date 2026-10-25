
import random
import sys

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
TRACE_FILE = sys.argv[2] if len(sys.argv) > 2 else 'Part2_Trace.txt'

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

alpha = 0.1
gamma = 0.99
epsilon = 0.2
theta = 0.001
minimum_visits = 100
quiet_episodes_required = 5
max_episodes = 300000

random.seed(SEED)

Q = {s: {a: 0.0 for a in transitions[s]} for s in transitions}
visits = {s: {a: 0 for a in transitions[s]} for s in transitions}

episode = 0
quiet_streak = 0
converged = False

with open(TRACE_FILE, 'w', encoding='utf-8') as trace:
    while episode < max_episodes:
        episode += 1
        state = 'RU_8p'
        episode_change = 0.0

        print(f'\nEPISODE {episode}', file=trace)

        while state != 'Terminal':
            actions = list(Q[state])

            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = max(Q[state], key=Q[state].get)

            outcomes = transitions[state][action]
            _, reward, next_state = random.choices(
                outcomes,
                weights=[item[0] for item in outcomes],
                k=1
            )[0]

            next_q = max(Q[next_state].values()) if next_state != 'Terminal' else 0.0
            previous_q = Q[state][action]
            new_q = previous_q + alpha * (reward + gamma * next_q - previous_q)

            Q[state][action] = new_q
            visits[state][action] += 1
            episode_change = max(episode_change, abs(new_q - previous_q))

            print(
                f'State: {state:<8} | Action: {action:<3} | '
                f'Previous Q: {previous_q:8.4f} | New Q: {new_q:8.4f} | '
                f'Reward: {reward:+d} | Next State: {next_state:<8} | '
                f'Next State Q: {next_q:8.4f}',
                file=trace
            )

            state = next_state

        fully_explored = all(
            count >= minimum_visits
            for state_visits in visits.values()
            for count in state_visits.values()
        )

        if fully_explored and episode_change < theta:
            quiet_streak += 1
        else:
            quiet_streak = 0

        if quiet_streak >= quiet_episodes_required:
            converged = True
            break

    results = [
        'FINAL RESULTS',
        f'Total Episodes: {episode}',
        '',
        'Final Q Values:'
    ]

    for s in Q:
        for a, value in Q[s].items():
            results.append(f'{s:<10} | {a:<3} : {value:.4f}')

    results.append('')
    results.append('Final Optimal Policy:')

    for s in states:
        if s == 'Terminal':
            results.append(f'{s:<10} -> No Action')
        else:
            results.append(f'{s:<10} -> {max(Q[s], key=Q[s].get)}')

    if not converged:
        results.append('Stopped at maximum episodes before reaching the stopping criterion.')

    for line in results:
        print(line)
        print(line, file=trace)
