# Author : Abhyanand Sharma, 24m0795
# Task 3 Agent


from minichess.chess.fastchess import Chess
from .base_agent import BaseAgent
import random
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves

PIECE_VALUE = {
    0: 100,
    1: 300,
    2: 350,
    3: 500,
    4: 900,
    5: 9000
}

PAWN_TABLE=[[0,0,0,0],
[5,10,10,5],
[10,20,20,10],
[20,40,40,20],
[0,0,0,0]]

KNIGHT_TABLE=[[-10,0,0,-10],
[0,10,10,0],
[5,20,20,5],
[5,15,15,5],
[-10,0,0,-10]]

BISHOP_TABLE=[[0,5,5,0],
[5,10,10,5],
[5,15,15,5],
[5,10,10,5],
[0,5,5,0]]

ROOK_TABLE=[[0,5,5,0],
[10,15,15,10],
[5,10,10,5],
[5,10,10,5],
[0,5,5,0]]

QUEEN_TABLE=[[0,5,5,0],
[5,10,10,5],
[5,15,15,5],
[5,10,10,5],
[0,5,5,0]]

KING_TABLE=[[20,30,30,20],
[10,15,15,10],
[0,0,0,0],
[-10,-10,-10,-10],
[-20,-30,-30,-20]]

PIECE_SQUARE_TABLES = {
    0: PAWN_TABLE,
    1: KNIGHT_TABLE,
    2: BISHOP_TABLE,
    3: ROOK_TABLE,
    4: QUEEN_TABLE,
    5: KING_TABLE
}

MAX_DEPTH = 3 

class Task3Agent(BaseAgent):
    def __init__(self, name="Task3Agent"):
        super().__init__(name)

    def move(self, chess_obj: Chess):
        piece_matrix, promotion_matrix = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(piece_matrix, promotion_matrix)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]
        best_move = None
        best_score = -10**9 if chess_obj.turn == 1 else 10**9
        ordered_moves = self.order_moves(legal_moves, chess_obj)
        for move in ordered_moves:
            (i, j), (dx, dy), promotion = move
            board_copy = chess_obj.copy() 
            board_copy.make_move(i, j, dx, dy, promotion)
            score = self.minimax(board_copy, MAX_DEPTH - 1, -10**9, 10**9, maximizing=(board_copy.turn == 1))
            if chess_obj.turn == 1:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move                    
        if best_move is None:
            best_move = random.choice(legal_moves)
        return best_move

    def minimax(self,chess_obj:Chess,depth:int,alpha:float,beta:float,maximizing:bool):
        result=chess_obj.game_result()
        if result is not None:
            if result == 1 : return 100000
            elif result == -1 : return -100000
            else : return -300
        if depth == 0 : return self.qsearch(chess_obj,alpha,beta,maximizing,q_depth=1)
        piece_matrix,promotion_matrix=chess_obj.legal_moves()
        moves = piece_matrix_to_legal_moves(piece_matrix,promotion_matrix)
        if not moves : return -300
        moves=self.order_moves(moves,chess_obj)
        if maximizing:
            max_eval = -10**9
            for move in moves:
                (i,j),(dx,dy),p =move
                b=chess_obj.copy()
                b.make_move(i,j,dx,dy,p)
                s=self.minimax(b,depth-1,alpha,beta,False)
                if s > max_eval : max_eval=s
                if s > alpha : alpha = s
                if beta<=alpha : break
            return max_eval
        else:
            min_eval = 10**9
            for move in moves:
                (i,j),(dx,dy),p=move
                b=chess_obj.copy()
                b.make_move(i,j,dx,dy,p)
                s=self.minimax(b,depth-1,alpha,beta,True)
                if s < min_eval: min_eval =s
                if s < beta: beta =s
                if beta <=alpha : break
            return min_eval

    def qsearch(self,chess_obj:Chess,alpha:float,beta:float,maximizing:bool,q_depth:int):
        s=self.eval(chess_obj)
        if q_depth == 0 : return s
        if maximizing : alpha=max(alpha,s)
        else: beta = min(beta,s)
        if beta <= alpha :return s
        pm,pr = chess_obj.legal_moves()
        moves = piece_matrix_to_legal_moves(pm,pr)
        moves = self.oco(moves,chess_obj)
        if not moves:return s
        for m in moves:
            (i,j),(dx,dy),p=m
            b = chess_obj.copy()
            b.make_move(i,j,dx,dy,p)
            sc = self.qsearch(b,alpha,beta,not maximizing,q_depth-1)
            if maximizing: alpha = max(alpha,sc)
            else: beta = min(beta,sc)
            if beta <= alpha : break
        return alpha if maximizing else beta

    def eval(self,chess_obj:Chess)->int:
        m=0;b=0
        for r in range(5):
            for c in range(4):
                p=chess_obj.any_piece_at(r,c)
                if not isinstance(p,tuple) or p[0]==-1 :continue
                t,col=p
                v=PIECE_VALUE.get(t,0)
                tbl=PIECE_SQUARE_TABLES[t]
                if col==1:
                    m+=v;b+=tbl[r][c]
                else:
                    m-=v;b-=tbl[4-r][c]
        return m+b

    def oco(self,moves,chess_obj:Chess):
        s=[]
        for m in moves:
            (i,j),(dx,dy),p=m
            ti=i+dx;tj=j+dy
            d=chess_obj.any_piece_at(ti,tj)
            cap=isinstance(d,tuple) and d[0]!=-1
            prom=p and p!=-1
            if not cap and not prom:continue
            sc=0
            if cap:sc+=PIECE_VALUE.get(d[0],0)*10
            if prom:sc+=5000
            s.append((sc,m))
        s.sort(key=lambda x:-x[0])
        return[m for sc,m in s]

    def order_moves(self,moves,chess_obj:Chess):
        s=[]
        for m in moves:
            (i,j),(dx,dy),p=m
            ti=i+dx;tj=j+dy
            d=chess_obj.any_piece_at(ti,tj)
            sc=0
            if isinstance(d,tuple) and d[0]!=-1:sc+=PIECE_VALUE.get(d[0],0)*10
            if p and p!=-1:sc+=5000
            s.append((sc,m))
        s.sort(key=lambda x:-x[0])
        return[m for sc,m in s]

    def reset(self,):
        ...