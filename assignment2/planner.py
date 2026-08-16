import argparse
import sys
import numpy as np
from pulp import LpProblem, LpVariable, lpSum, PULP_CBC_CMD, LpMinimize

def parse_mdp(filepath):
    content = None
    for encoding in ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be']:
        try:
            with open(filepath,"r",encoding=encoding) as f:
                content = f.read()
            break
        except Exception:
            continue
    if content is None:
        sys.exit(1)
    lines = content.strip().split('\n')
    num_states,num_actions,gamma,mdptype = 0,0,0.0,""
    terminal_states = set()
    transition_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split()
        keyword = parts[0]
        if keyword == "numStates": num_states = int(parts[1])
        elif keyword == "numActions": num_actions = int(parts[1])
        elif keyword == "end": terminal_states.update(map(int,parts[1:]))
        elif keyword == "transition": transition_lines.append(parts[1:])
        elif keyword == "mdptype": mdptype = parts[1]
        elif keyword == "discount": gamma = float(parts[1])
    transitions = {s:{} for s in range(num_states)}
    for parts in transition_lines:
        s1,a,s2 = int(parts[0]),int(parts[1]),int(parts[2])
        r,p = float(parts[3]),float(parts[4])
        if a not in transitions[s1]:transitions[s1][a] = []
        transitions[s1][a].append((s2,r,p))
    return {"num_states": num_states,"num_actions": num_actions,"transitions": transitions,
            "terminal_states": terminal_states,"gamma": gamma,"mdptype": mdptype}

def evaluate_policyI(mdp,policy,max_iter=10000,tol=1e-9):
    S,gamma,transitions,terminal_states = mdp["num_states"],mdp["gamma"],mdp["transitions"],mdp["terminal_states"]
    V = np.zeros(S)
    for _ in range(max_iter):
        delta = 0
        V_old = np.copy(V)
        for s in range(S):
            if s in terminal_states: continue
            a = policy[s]
            v_s = 0.0
            if a != -1 and a in transitions.get(s, {}):
                v_s = sum(p*(r+gamma*V_old[s_next]) for s_next,r,p in transitions[s][a])
            delta = max(delta,abs(v_s-V[s]))
            V[s] = v_s
        if delta<tol:break
    return V

def evaluate_policyF(mdp,policy,V_old,k=10):
    S,gamma,transitions,terminal_states = mdp["num_states"],mdp["gamma"],mdp["transitions"],mdp["terminal_states"]
    V = np.copy(V_old)
    for _ in range(k):
        V_k = np.copy(V)
        for s in range(S):
            if s in terminal_states: continue
            a = policy[s]
            v_s = 0.0
            if a != -1 and a in transitions.get(s, {}):
                v_s = sum(p*(r+gamma*V_k[s_next]) for s_next,r,p in transitions[s][a])
            V[s] = v_s
    return V

def hpi(mdp,max_iter=1000):
    S,gamma,transitions,terminal_states = mdp["num_states"],mdp["gamma"],mdp["transitions"],mdp["terminal_states"]
    policy = np.zeros(S,dtype=int)
    for s in terminal_states:policy[s] = -1
    V = np.zeros(S)
    for _ in range(max_iter):
        V = evaluate_policyF(mdp,policy,V)
        policy_stable = True
        for s in range(S):
            if s in terminal_states:continue
            old_action = policy[s]
            available_actions = transitions.get(s,{})
            if not available_actions:continue
            q_values = {a:sum(p*(r+gamma*V[s_next]) for s_next,r,p in available_actions[a]) for a in available_actions}
            best_a = max(q_values,key=q_values.get)
            if best_a != old_action:
                policy[s] = best_a
                policy_stable = False
        if policy_stable: break
    V_final = evaluate_policyI(mdp,policy)
    return V_final, policy

def lp(mdp):
    S,gamma,transitions,terminal_states = mdp["num_states"],mdp["gamma"],mdp["transitions"],mdp["terminal_states"]
    prob = LpProblem("MDP_LP",LpMinimize)
    V_vars = [LpVariable(f"V_{s}") for s in range(S)]
    prob += lpSum(V_vars)
    for s in range(S):
        if s in terminal_states:
            prob += V_vars[s] == 0
            continue
        for a in transitions.get(s, {}):
            expected_val = lpSum([p*(r+gamma*V_vars[s_next]) for s_next,r,p in transitions[s][a]])
            prob += V_vars[s] >= expected_val

    prob.solve(PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        print("LP solver failed.",file=sys.stderr)
        return np.zeros(S),np.zeros(S,dtype=int)
    
    V_opt = np.array([v.varValue for v in V_vars])
    policy = np.zeros(S,dtype=int)
    for s in range(S):
        if s in terminal_states:
            policy[s] = -1
            continue
        available_actions = transitions.get(s,{})
        if not available_actions:
            policy[s] = -1
            continue
        q_values = {a:sum(p*(r+gamma*V_opt[s_next]) for s_next,r,p in available_actions[a]) for a in available_actions}
        policy[s] = max(q_values,key=q_values.get)
    return V_opt, policy

def print_results(V, policy):
    for i in range(len(V)):
        print(f"{V[i]:.6f}\t{policy[i]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp",required=True)
    parser.add_argument("--algorithm",choices=["hpi","lp"],default="hpi")
    parser.add_argument("--policy",required=False)
    args = parser.parse_args()    
    mdp = parse_mdp(args.mdp)
    if args.policy:
        try:
            policy_from_file = np.loadtxt(args.policy,dtype=int)
            policy = policy_from_file if len(policy_from_file.shape) == 1 else policy_from_file[:,1]
            V = evaluate_policyI(mdp,policy)
            print_results(V,policy)
        except Exception as e:
            print(f"Error loading or evaluating policy: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if args.algorithm == "lp":
            V, policy = lp(mdp)
        else:
            V, policy = hpi(mdp)
        print_results(V, policy)

if __name__ == "__main__":
    main()