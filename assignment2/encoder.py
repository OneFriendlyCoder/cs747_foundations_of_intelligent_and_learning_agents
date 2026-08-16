import argparse
from collections import Counter, deque, defaultdict

def parse_game_config(path):
    with open(path,"r",encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    threshold = int(lines[1])
    bonus = int(lines[2])
    special_seq = set(map(int,lines[3].split())) if len(lines) > 3 else set()
    return threshold, bonus, special_seq

def fullDeck():
    return [f"{n}{s}" for n in range(1,14) for s in ("H","D")]

def card_value(card):
    return int(card[:-1])

def hand_value(hand):
    return sum(card_value(c) for c in hand)

def special_sequence(hand,special_seq):
    if not special_seq: return False
    hand_values = {card_value(c) for c in hand}
    return special_seq.issubset(hand_values)

def actionid4card(card):
    num = card_value(card)
    suit = card[-1]
    return num if suit == "H" else num+13

def encode_mdp(game_config):
    threshold, bonus, special_seq = parse_game_config(game_config)
    full_set = set(fullDeck())
    terminal_key = "TERMINAL"
    start_key = ()
    q = deque([start_key])
    seen = {start_key}
    transitions_tmp = []
    while q:
        hand = q.popleft()
        hv = hand_value(hand)
        if hv >= threshold:
            continue
        remaining_cards = list(full_set-set(hand))
        if remaining_cards:
            prob = 1.0 / len(remaining_cards)
            for draw_card in remaining_cards:
                new_hand = tuple(sorted(list(hand)+[draw_card]))
                if hand_value(new_hand) >= threshold:
                    transitions_tmp.append((hand,0,terminal_key,0.0,prob))
                else:
                    transitions_tmp.append((hand,0,new_hand,0.0,prob))
                    if new_hand not in seen:
                        seen.add(new_hand)
                        q.append(new_hand)
        if hand and remaining_cards:
            prob = 1.0/len(remaining_cards)
            for card_in_hand in hand:
                action = actionid4card(card_in_hand)
                for draw_card in remaining_cards:
                    new_hand_list = list(hand)
                    new_hand_list.remove(card_in_hand)
                    new_hand_list.append(draw_card)
                    new_hand = tuple(sorted(new_hand_list))
                    if hand_value(new_hand) >= threshold:
                        transitions_tmp.append((hand,action,terminal_key,0.0,prob))
                    else:
                        transitions_tmp.append((hand,action,new_hand,0.0,prob))
                        if new_hand not in seen:
                            seen.add(new_hand)
                            q.append(new_hand)
        final_score = float(hv+(bonus if special_sequence(hand,special_seq) else 0))
        transitions_tmp.append((hand,27,terminal_key,final_score,1.0))

    non_terminal_states = sorted(list(seen),key=lambda h:(len(h),h))
    state_to_id = {terminal_key:0,**{s:i+1 for i, s in enumerate(non_terminal_states)}}

    agg = defaultdict(lambda:[0.0, 0.0])
    for s_key,a,t_key,r,p in transitions_tmp:
        s_id, t_id = state_to_id[s_key], state_to_id[t_key]
        agg[(s_id,a,t_id)][0] += p
        agg[(s_id,a,t_id)][1] = r

    print(f"numStates {len(state_to_id)}")
    print("numActions 28")
    print("end 0")
    for (s, a, t), (p, r) in sorted(agg.items()):
        if p > 1e-9:
            print(f"transition {s} {a} {t} {r:.6f} {p:.6f}")        
    print("mdptype episodic")
    print("discount 1.0")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_config",required=True)
    args = parser.parse_args()
    encode_mdp(args.game_config)

if __name__ == "__main__":
    main()