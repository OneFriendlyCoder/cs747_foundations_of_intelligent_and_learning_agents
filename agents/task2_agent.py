# Author : Abhyanand Sharma, 24m0795
# Task 2 Agent


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
class Task2Agent(BaseAgent):
    def __init__(self, name="Task2Agent"):
        super().__init__(name)

    def move(self, chess_obj:Chess):
        piece_matrix,promotion_matrix = chess_obj.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(piece_matrix,promotion_matrix)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        best_move = None
        best_score = -10**9 if chess_obj.turn == 1 else 10**9
        ordered_moves = self.omf(legal_moves,chess_obj)
        
        for move in ordered_moves:
            (i,j),(dx,dy),promotion = move
            board_copy = chess_obj.copy()
            board_copy.make_move(i,j,dx,dy,promotion)
            score = self.minimax(board_copy,MAX_DEPTH-1,-10**9,10**9,maximizing=(board_copy.turn == 1))
            
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
        result = chess_obj.game_result()
        if result is not None:
            if result == 1:
                return 100000
            elif result == -1:
                return -100000
            else:
                return 0
        if depth == 0:
            return self.eval(chess_obj)
        piece_matrix,promotion_matrix = chess_obj.legal_moves()
        moves = piece_matrix_to_legal_moves(piece_matrix,promotion_matrix)
        if not moves:
            return 0
        moves = self.omf(moves,chess_obj)
        
        if maximizing:
            max_eval = -10**9
            for move in moves:
                (i,j),(dx,dy),promotion = move
                new_board = chess_obj.copy()
                new_board.make_move(i,j,dx,dy,promotion)
                eval_score = self.minimax(new_board,depth-1,alpha,beta,False)
                max_eval = max(max_eval,eval_score)
                alpha = max(alpha,eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = 10**9
            for move in moves:
                (i,j),(dx,dy),promotion = move
                new_board = chess_obj.copy()
                new_board.make_move(i,j,dx,dy,promotion)
                eval_score = self.minimax(new_board,depth-1,alpha,beta,True)
                min_eval = min(min_eval,eval_score)
                beta = min(beta,eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def eval(self,chess_obj:Chess) -> int:
        material = 0
        piece_square_bonus = 0
        for r in range(5):
            for c in range(4):
                piece = chess_obj.any_piece_at(r, c)
                if not isinstance(piece, tuple) or piece[0] == -1:
                    continue
                piece_type, color = piece
                val = PIECE_VALUE[piece_type]
                table = PIECE_SQUARE_TABLES[piece_type]
                if color == 1:
                    material += val
                    piece_square_bonus += table[r][c]
                else:
                    material -= val
                    piece_square_bonus -= table[4-r][c]
        return material+piece_square_bonus

    def ocof(self,moves,chess_obj:Chess):
        captures=[]
        for move in moves:
            (i,j),(dx,dy),promotion=move
            to_i=i+dx
            to_j=j+dy
            dest=chess_obj.any_piece_at(to_i,to_j)
            is_capture=isinstance(dest,tuple)and dest[0]!=-1
            is_promotion=promotion and promotion!=-1
            if not is_capture and not is_promotion:continue
            score=0
            if is_capture:score+=PIECE_VALUE[dest[0]]*10
            if is_promotion:score+=5000
            captures.append((score,move))
        captures.sort(reverse=True)
        return [m for _,m in captures]

    def omf(self,moves,chess_obj:Chess):
        scored=[]
        for move in moves:
            (i,j),(dx,dy),promotion=move
            to_i=i+dx
            to_j=j+dy
            dest=chess_obj.any_piece_at(to_i,to_j)
            score=0          
            if isinstance(dest,tuple) and dest[0] != -1:
                score += PIECE_VALUE[dest[0]] * 10
            if promotion and promotion != -1:
                score += 5000
            scored.append((score,move))
        scored.sort(reverse=True)
        return [m for _, m in scored]

    def reset(self):
        pass


