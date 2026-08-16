
# Author : Abhyanand Sharma, 24m0795
# Task 4 Agent

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

MAX_DEPTH = 5
MAX_Q_DEPTH = 5
WIN_VALUE = 100000
DRAW_AS_LOSS_SCORE = -WIN_VALUE + 1000
DRAW_AS_WIN_SCORE = WIN_VALUE - 1000
NEUTRAL_DRAW_SCORE = -300
MATERIAL_ADVANTAGE_THRESHOLD = 150
MOBILITY_WEIGHT = 3
DELTA_PRUNING_MARGIN = 350
DOUBLED_PAWN_PENALTY = -10
ROOK_SEMI_OPEN_FILE_BONUS = 10
ROOK_OPEN_FILE_BONUS = 15
TT_EXACT = 0
TT_ALPHA = 1
TT_BETA = 2

class Task4Agent(BaseAgent):
    def __init__(self,name="Task4Agent"):
        super().__init__(name)
        self.transposition_table={}

    def move(self,chess_obj:Chess):
        self.reset()
        piece_matrix,promotion_matrix=chess_obj.legal_moves()
        legal_moves=piece_matrix_to_legal_moves(piece_matrix,promotion_matrix)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]
        best_move=None
        best_score=-float('inf') if chess_obj.turn == 1 else float('inf')
        ordered_moves=self.order_moves(legal_moves,chess_obj)
        alpha=-float('inf')
        beta=float('inf')
        for move in ordered_moves:
            (i,j),(dx,dy),promotion=move
            board_copy=chess_obj.copy()
            board_copy.make_move(i,j,dx,dy,promotion)
            score=self.minimax(board_copy,MAX_DEPTH-1,alpha,beta,(board_copy.turn == 1))
            if chess_obj.turn == 1:
                if score>best_score:
                    best_score=score
                    best_move=move
                alpha=max(alpha,best_score)
            else:
                if score<best_score:
                    best_score=score
                    best_move=move
                beta=min(beta,best_score)
        if best_move is None:
            best_move=random.choice(legal_moves)
        return best_move

    def _get_board_key(self,chess_obj:Chess):
        board_state=[]
        for r in range(5):
            row_state=[]
            for c in range(4):
                row_state.append(chess_obj.any_piece_at(r,c))
            board_state.append(tuple(row_state))
        return (tuple(board_state),chess_obj.turn)

    def _get_draw_score(self,material_balance:int)->int:
        if material_balance>MATERIAL_ADVANTAGE_THRESHOLD:
            return DRAW_AS_LOSS_SCORE
        elif material_balance<-MATERIAL_ADVANTAGE_THRESHOLD:
            return DRAW_AS_WIN_SCORE
        else:
            return NEUTRAL_DRAW_SCORE

    def minimax(self,chess_obj:Chess,depth:int,alpha:float,beta:float,maximizing:bool):
        board_key=self._get_board_key(chess_obj)
        tt_entry=self.transposition_table.get(board_key)
        if tt_entry is not None:
            tt_score,tt_depth,tt_flag=tt_entry
            if tt_depth>=depth:
                if tt_flag == TT_EXACT:
                    return tt_score
                elif tt_flag == TT_ALPHA:
                    alpha=max(alpha,tt_score)
                elif tt_flag == TT_BETA:
                    beta=min(beta,tt_score)
                if beta<=alpha:
                    return tt_score
        result=chess_obj.game_result()
        if result is not None:
            if result == 1:
                return WIN_VALUE
            elif result == -1:
                return -WIN_VALUE
            else:
                _,material_balance=self.eval(chess_obj)
                return self._get_draw_score(material_balance)
        if depth == 0:
            return self.qsearch(chess_obj,alpha,beta,maximizing,MAX_Q_DEPTH)
        piece_matrix,promotion_matrix=chess_obj.legal_moves()
        moves=piece_matrix_to_legal_moves(piece_matrix,promotion_matrix)
        if not moves:
            _,material_balance=self.eval(chess_obj)
            return self._get_draw_score(material_balance)
        moves=self.order_moves(moves,chess_obj)
        tt_flag=TT_ALPHA
        if maximizing:
            max_eval=-float('inf')
            for move in moves:
                (i,j),(dx,dy),promotion=move
                new_board=chess_obj.copy()
                new_board.make_move(i,j,dx,dy,promotion)
                eval_score=self.minimax(new_board,depth-1,alpha,beta,False)
                max_eval=max(max_eval,eval_score)
                if max_eval>alpha:
                    alpha=max_eval
                    tt_flag=TT_EXACT
                if beta<=alpha:
                    self.transposition_table[board_key]=(max_eval,depth,TT_BETA)
                    return max_eval
            self.transposition_table[board_key]=(max_eval,depth,tt_flag)
            return max_eval
        else:
            min_eval=float('inf')
            for move in moves:
                (i,j),(dx,dy),promotion=move
                new_board=chess_obj.copy()
                new_board.make_move(i,j,dx,dy,promotion)
                eval_score=self.minimax(new_board,depth-1,alpha,beta,True)
                min_eval=min(min_eval,eval_score)
                if min_eval<beta:
                    beta=min_eval
                    tt_flag=TT_EXACT
                if beta<=alpha:
                    self.transposition_table[board_key]=(min_eval,depth,TT_ALPHA)
                    return min_eval
            self.transposition_table[board_key]=(min_eval,depth,tt_flag)
            return min_eval

    def qsearch(self,chess_obj:Chess,alpha:float,beta:float,maximizing:bool,q_depth:int):
        result=chess_obj.game_result()
        if result is not None:
            if result == 1:
                return WIN_VALUE
            elif result == -1:
                return -WIN_VALUE
            else:
                _,material_balance=self.eval(chess_obj)
                return self._get_draw_score(material_balance)
        stand_pat_score,_=self.eval(chess_obj)
        if q_depth == 0:
            return stand_pat_score
        if maximizing:
            if stand_pat_score>=beta:
                return stand_pat_score
            alpha=max(alpha,stand_pat_score)
        else:
            if stand_pat_score<=alpha:
                return stand_pat_score
            beta=min(beta,stand_pat_score)
        piece_matrix,promotion_matrix=chess_obj.legal_moves()
        moves=piece_matrix_to_legal_moves(piece_matrix,promotion_matrix)
        if not moves:
            _,material_balance=self.eval(chess_obj)
            return self._get_draw_score(material_balance)
        capture_moves=self.oco(moves,chess_obj)
        if not capture_moves:
            return stand_pat_score
        if maximizing:
            best_score=stand_pat_score
            for move in capture_moves:
                (i,j),(dx,dy),promotion=move
                dest=chess_obj.any_piece_at(i+dx,j+dy)
                if isinstance(dest,tuple) and dest[0]!=-1:
                    gain=PIECE_VALUE.get(dest[0],0)
                    if stand_pat_score+gain+DELTA_PRUNING_MARGIN<alpha:
                        continue
                new_board=chess_obj.copy()
                new_board.make_move(i,j,dx,dy,promotion)
                score=self.qsearch(new_board,alpha,beta,False,q_depth-1)
                best_score=max(best_score,score)
                alpha=max(alpha,score)
                if beta<=alpha:
                    break
            return best_score
        else:
            best_score=stand_pat_score
            for move in capture_moves:
                (i,j),(dx,dy),promotion=move
                dest=chess_obj.any_piece_at(i+dx,j+dy)
                if isinstance(dest,tuple) and dest[0]!=-1:
                    gain=PIECE_VALUE.get(dest[0],0)
                    if stand_pat_score-gain-DELTA_PRUNING_MARGIN>beta:
                        continue
                new_board=chess_obj.copy()
                new_board.make_move(i,j,dx,dy,promotion)
                score=self.qsearch(new_board,alpha,beta,True,q_depth-1)
                best_score=min(best_score,score)
                beta=min(beta,score)
                if beta<=alpha:
                    break
            return best_score

    def eval(self,chess_obj:Chess)->(int,int):
        material_balance=0
        piece_square_bonus=0
        structure_bonus=0
        white_pawn_cols=set()
        black_pawn_cols=set()
        white_rook_cols=[]
        black_rook_cols=[]
        for r in range(5):
            for c in range(4):
                piece=chess_obj.any_piece_at(r,c)
                if not isinstance(piece,tuple) or piece[0] == -1:
                    continue
                piece_type,color=piece
                val=PIECE_VALUE.get(piece_type,0)
                table=PIECE_SQUARE_TABLES[piece_type]
                if color == 1:
                    material_balance += val
                    piece_square_bonus += table[r][c]
                    if piece_type == 0:
                        if c in white_pawn_cols:
                            structure_bonus += DOUBLED_PAWN_PENALTY
                        white_pawn_cols.add(c)
                    elif piece_type == 3:
                        white_rook_cols.append(c)
                else:
                    material_balance -= val
                    piece_square_bonus -= table[4-r][c]
                    if piece_type == 0:
                        if c in black_pawn_cols:
                            structure_bonus -= DOUBLED_PAWN_PENALTY
                        black_pawn_cols.add(c)
                    elif piece_type == 3:
                        black_rook_cols.append(c)
        for c in white_rook_cols:
            if c not in white_pawn_cols:
                structure_bonus += ROOK_SEMI_OPEN_FILE_BONUS
                if c not in black_pawn_cols:
                    structure_bonus += ROOK_OPEN_FILE_BONUS
        for c in black_rook_cols:
            if c not in black_pawn_cols:
                structure_bonus -= ROOK_SEMI_OPEN_FILE_BONUS
                if c not in white_pawn_cols:
                    structure_bonus -= ROOK_OPEN_FILE_BONUS
        piece_matrix,promotion_matrix=chess_obj.legal_moves()
        moves=piece_matrix_to_legal_moves(piece_matrix,promotion_matrix)
        mobility_bonus=len(moves)*MOBILITY_WEIGHT
        if chess_obj.turn == -1:
            mobility_bonus=-mobility_bonus
        total_score=material_balance+piece_square_bonus+structure_bonus+mobility_bonus
        return total_score,material_balance

    def oco(self,moves,chess_obj:Chess):
        scored=[]
        for move in moves:
            (i,j),(dx,dy),promotion=move
            to_i=i+dx
            to_j=j+dy
            dest=chess_obj.any_piece_at(to_i,to_j)
            source=chess_obj.any_piece_at(i,j)
            is_capture=isinstance(dest,tuple) and dest[0]!=-1
            is_promotion=promotion and promotion!=-1
            if not is_capture and not is_promotion:
                continue
            score=0
            if is_promotion:
                score += 20000+PIECE_VALUE.get(promotion,PIECE_VALUE[4])
            if is_capture:
                captured_type=dest[0]
                attacker_type=source[0] if isinstance(source,tuple) else 0
                score += 10000+(PIECE_VALUE.get(captured_type,0)*10-PIECE_VALUE.get(attacker_type,0))
            scored.append((score,move))
        scored.sort(key=lambda x:-x[0])
        return[m for s,m in scored]

    def order_moves(self,moves,chess_obj:Chess):
        scored=[]
        for move in moves:
            (i,j),(dx,dy),promotion=move
            to_i=i+dx
            to_j=j+dy
            dest=chess_obj.any_piece_at(to_i,to_j)
            source=chess_obj.any_piece_at(i,j)
            score=0
            is_capture=isinstance(dest,tuple) and dest[0]!=-1
            if promotion and promotion!=-1:
                score += 20000+PIECE_VALUE.get(promotion,PIECE_VALUE[4])
            if is_capture:
                captured_type=dest[0]
                attacker_type=source[0] if isinstance(source,tuple) else 0
                score += 10000+(PIECE_VALUE.get(captured_type,0)*10-PIECE_VALUE.get(attacker_type,0))
            else:
                if isinstance(source,tuple):
                    source_type,color=source
                    table=PIECE_SQUARE_TABLES[source_type]
                    current_sq_val=table[i][j] if color == 1 else table[4-i][j]
                    next_sq_val=table[to_i][to_j] if color == 1 else table[4-to_i][to_j]
                    score += next_sq_val-current_sq_val
            scored.append((score,move))
        scored.sort(key=lambda x:-x[0])
        return[m for s,m in scored]

    def reset(self):
        self.transposition_table={}
