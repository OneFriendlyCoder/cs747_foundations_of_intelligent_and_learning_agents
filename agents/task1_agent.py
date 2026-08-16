# Author : Abhyanand Sharma, 24m0795
# Task 1 Agent


from minichess.chess.fastchess import Chess
from .base_agent import BaseAgent
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves
import random

class Task1Agent(BaseAgent):
    def __init__(self,name="Task1Agent"):
        super().__init__(name)
        self.piece_values=[100,320,330,500,900,20000]
        self.winning_threshold=600
        self.losing_threshold=-600
        self.max_opp_samples=2

    def move(self,chess_obj:Chess):
        pm,pr=chess_obj.legal_moves()
        moves=piece_matrix_to_legal_moves(pm,pr)
        if not moves:return None
        root_turn=chess_obj.turn
        best_move=moves[0]
        best_score=-float('inf')
        random.shuffle(moves)
        for(i,j),(dx,dy),promo in moves:
            s1=chess_obj.copy()
            s1.make_move(i,j,dx,dy,promo)
            score=self.feval(s1,root_turn)
            if score>best_score:
                best_score=score
                best_move=((i,j),(dx,dy),promo)
        return best_move

    def deval(self,state:Chess,root_turn:int):
        if not state.has_legal_moves:
            r=state.game_result()
            if(r==1 and root_turn==1)or(r==-1 and root_turn==0):return 99999
            elif(r==-1 and root_turn==1)or(r==1 and root_turn==0):return -99999
            return 0
        score=0
        for i in range(5):
            for j in range(4):
                t,c=state.any_piece_at(i,j)
                if t!=-1:
                    v=self.piece_values[t]
                    if c==1:
                        score+=v
                        if t==0:
                            if i==4:score+=500
                            elif i==3:score+=200
                            else:score+=i*30
                    else:
                        score-=v
                        if t==0:
                            if i==0:score-=500
                            elif i==1:score-=200
                            else:score-=(4-i)*30
        return score if root_turn==1 else -score

    def nogo_squares(self,state:Chess):
        pm,pr=state.legal_moves()
        moves=piece_matrix_to_legal_moves(pm,pr)
        return {(i+dx,j+dy)for(i,j),(dx,dy),_ in moves}

    def feval(self,s1,root_turn):
        pm2,pr2=s1.legal_moves()
        opp_moves=piece_matrix_to_legal_moves(pm2,pr2)
        if not opp_moves:return self.deval(s1,root_turn)
        threatened=self.nogo_squares(s1)
        score=self.meval(s1,root_turn)
        for i in range(5):
            for j in range(4):
                t,c=s1.any_piece_at(i,j)
                if t!=-1 and c==root_turn and(i,j)in threatened:
                    score-=self.piece_values[t]*0.8
        return score

    def meval(self,state:Chess,root_turn:int):
        s=0
        for i in range(5):
            for j in range(4):
                t,c=state.any_piece_at(i,j)
                if t!=-1:
                    v=self.piece_values[t]
                    s+=v if c==1 else -v
        return s if root_turn==1 else -s

    def reset(self):
        pass
