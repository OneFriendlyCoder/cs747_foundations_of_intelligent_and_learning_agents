import argparse
from collections import deque

def fullDeck():
    return[f"{n}{s}" for n in range(1, 14) for s in ("H","D")]

def card_value(card):
    return int(card[:-1])

def hand_value(hand):
    return sum(card_value(c) for c in hand)

def genStateMap(thres):
    fs = set(fullDeck())
    sk = ()
    q = deque([sk])
    seen = {sk}    
    while q:
        hand = q.popleft()
        if hand_value(hand) >= thres: 
            continue
        remaining_cards =list(fs-set(hand))        
        if remaining_cards:
            for c in remaining_cards:
                new_hand = tuple(sorted(list(hand)+[c]))
                if hand_value(new_hand)<thres and new_hand not in seen:
                    seen.add(new_hand)
                    q.append(new_hand)

        if hand and remaining_cards:
            for cih in hand:
                for c in remaining_cards:
                    new_hand_list = list(hand)
                    new_hand_list.remove(cih)
                    new_hand_list.append(c)
                    new_hand = tuple(sorted(new_hand_list))
                    if hand_value(new_hand)<thres and new_hand not in seen:
                        seen.add(new_hand)
                        q.append(new_hand)
                        
    nontstates = sorted(list(seen), key=lambda h: (len(h), h))
    return {hand:i+1 for i, hand in enumerate(nontstates)}

def parseTestcases(fp):
    with open(fp,"r",encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    thres=int(lines[1])
    hands=[]
    try:
        start = lines.index('Testcase:') + 1
    except ValueError:
        start = 4
    for line in lines[start:]:
        hands.append(tuple(sorted(line.split())))
    return thres,hands

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--value_policy", required=True)
    parser.add_argument("--testcase", required=True)
    args = parser.parse_args()
    thres,test_hands = parseTestcases(args.testcase)
    hand_to_id = genStateMap(thres)
    policy = {}
    with open(args.value_policy,'r') as f:
        for state_id,line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 2:
                policy[state_id] = int(parts[1])
    for hand in test_hands:
        state_id = hand_to_id.get(hand)
        action = policy.get(state_id, 27) if state_id is not None else 27
        print(action)

if __name__ == "__main__":
    main()