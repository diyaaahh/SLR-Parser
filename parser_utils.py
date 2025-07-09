from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd

# Type aliases for clarity
Production = Tuple[str, List[str]]
Grammar = List[Production]


def parse_grammar(grammar_text: str) -> Grammar:
    """
    Parses grammar from a string into a list of productions.
    Each production is a tuple (LHS, RHS_list).
    """
    grammar = []
    for line in grammar_text.strip().split('\n'):
        if '->' not in line:
            continue
        lhs, rhs = line.split('->')
        lhs = lhs.strip()
        rhs_alts = rhs.strip().split('|')
        for alt in rhs_alts:
            grammar.append((lhs, alt.strip().split()))
    return grammar


def compute_first_sets(grammar: Grammar) -> Dict[str, Set[str]]:
    """
    Computes the FIRST sets for all non-terminals in the grammar.
    Returns a dictionary: non-terminal -> set of terminals.
    """
    first = defaultdict(set)
    productions = defaultdict(list)
    non_terminals = set()
    terminals = set()

    # Collect non-terminals and terminals
    for lhs, rhs in grammar:
        productions[lhs].append(rhs)
        non_terminals.add(lhs)
        for symbol in rhs:
            if symbol not in non_terminals and not symbol.isupper():
                terminals.add(symbol)

    def first_of(symbol):
        if symbol in terminals:
            return {symbol}
        return first[symbol]

    changed = True
    while changed:
        changed = False
        for lhs, rhss in productions.items():
            for rhs in rhss:
                before = len(first[lhs])
                if not rhs:
                    # epsilon production
                    first[lhs].add('ε')
                else:
                    for symbol in rhs:
                        first[lhs].update(first_of(symbol) - {'ε'})
                        if 'ε' not in first_of(symbol):
                            break
                    else:
                        first[lhs].add('ε')
                if len(first[lhs]) > before:
                    changed = True
    return dict(first)


def compute_follow_sets(grammar: Grammar, first_sets: Dict[str, Set[str]], start_symbol: str) -> Dict[str, Set[str]]:
    """
    Computes the FOLLOW sets for all non-terminals in the grammar.
    Returns a dictionary: non-terminal -> set of terminals.
    """
    follow = defaultdict(set)
    productions = defaultdict(list)
    non_terminals = set()

    for lhs, rhs in grammar:
        productions[lhs].append(rhs)
        non_terminals.add(lhs)

    follow[start_symbol].add('$')  # End marker

    changed = True
    while changed:
        changed = False
        for lhs, rhss in productions.items():
            for rhs in rhss:
                for i, symbol in enumerate(rhs):
                    if symbol in non_terminals:
                        trailer = set()
                        # Look ahead in the production
                        for next_symbol in rhs[i+1:]:
                            if next_symbol in first_sets:
                                trailer = first_sets[next_symbol] - {'ε'}
                                epsilon_in_next = 'ε' in first_sets[next_symbol]
                            else:
                                trailer = {next_symbol}
                                epsilon_in_next = False
                            before = len(follow[symbol])
                            follow[symbol].update(trailer)
                            if epsilon_in_next:
                                continue
                            else:
                                break
                        else:
                            before = len(follow[symbol])
                            follow[symbol].update(follow[lhs])
                        if len(follow[symbol]) > before:
                            changed = True
    return dict(follow)


def closure(items: Set[Tuple[str, Tuple[str, ...], int]], grammar: Grammar) -> Set[Tuple[str, Tuple[str, ...], int]]:
    """
    Computes the closure of a set of LR(0) items for the given grammar.
    Each item is a tuple: (lhs, rhs tuple, dot position).
    """
    closure_set = set(items)
    productions = defaultdict(list)
    for lhs, rhs in grammar:
        productions[lhs].append(tuple(rhs))
    added = True
    while added:
        added = False
        new_items = set()
        for (lhs, rhs, dot_pos) in closure_set:
            if dot_pos < len(rhs):
                symbol = rhs[dot_pos]
                if symbol in productions:  # symbol is a non-terminal
                    for prod_rhs in productions[symbol]:
                        item = (symbol, prod_rhs, 0)
                        if item not in closure_set and item not in new_items:
                            new_items.add(item)
        if new_items:
            closure_set.update(new_items)
            added = True
    return closure_set

def goto(items: Set[Tuple[str, Tuple[str, ...], int]], symbol: str, grammar: Grammar) -> Set[Tuple[str, Tuple[str, ...], int]]:
    """
    Computes the GOTO set from a set of LR(0) items and a grammar symbol.
    """
    moved_items = set()
    for (lhs, rhs, dot_pos) in items:
        if dot_pos < len(rhs) and rhs[dot_pos] == symbol:
            moved_items.add((lhs, rhs, dot_pos + 1))
    # Only take closure of the moved items, not the original items
    return closure(moved_items, grammar) if moved_items else set()

def construct_lr0_item_sets(grammar: Grammar):
    """
    Constructs the canonical collection of LR(0) item sets for the grammar.
    Returns a list of item sets and a GOTO table: (item_sets, goto_table)
    """
    # Convert all RHS to tuples to ensure hashability
    grammar = [(lhs, tuple(rhs)) for lhs, rhs in grammar]
    
    # All grammar symbols
    symbols = set()
    for lhs, rhs in grammar:
        symbols.update(rhs)
        symbols.add(lhs)
    symbols = sorted(symbols)

    # Start with the augmented grammar's start production
    start_item = (grammar[0][0], grammar[0][1], 0)
    initial = closure({start_item}, grammar)
    item_sets = [initial]
    item_set_indices = {frozenset(initial): 0}
    goto_table = {}

    queue = [initial]
    while queue:
        I = queue.pop(0)
        I_idx = item_set_indices[frozenset(I)]
        for symbol in symbols:
            goto_I = goto(I, symbol, grammar)
            if goto_I:
                frozen_goto = frozenset(goto_I)
                if frozen_goto not in item_set_indices:
                    item_set_indices[frozen_goto] = len(item_sets)
                    item_sets.append(goto_I)
                    queue.append(goto_I)
                goto_table[(I_idx, symbol)] = item_set_indices[frozen_goto]

    return item_sets, goto_table

def build_slr_parsing_table(grammar: Grammar, item_sets, goto_table, follow_sets):
    """
    Constructs the SLR parsing table (ACTION and GOTO) for the given grammar.
    Returns two dicts: action_table and goto_table.
    action_table[(state, symbol)] = 'shift s', 'reduce r', 'accept', or 'error'
    goto_table[(state, nonterminal)] = state
    """
    # Map productions to numbers for reduce actions
    prod_list = []
    prod_map = {}
    for idx, (lhs, rhs) in enumerate(grammar):
        prod_list.append((lhs, tuple(rhs)))
        prod_map[(lhs, tuple(rhs))] = idx

    terminals = set()
    nonterminals = set()
    for lhs, rhs in grammar:
        nonterminals.add(lhs)
        for sym in rhs:
            if sym not in nonterminals and not sym.isupper():
                terminals.add(sym)
            if sym == 'id' or sym == '*':
                terminals.add(sym)
    terminals.add('$')

    action_table = {}
    goto_table_out = {}

    for i, item_set in enumerate(item_sets):
        for item in item_set:
            lhs, rhs, dot_pos = item
            # Shift
            if dot_pos < len(rhs):
                a = rhs[dot_pos]
                if a in terminals:
                    if (i, a) in goto_table:
                        action_table[(i, a)] = f'shift {goto_table[(i, a)]}'
            # Reduce or Accept
            else:
                if lhs == "S'":
                    action_table[(i, '$')] = 'accept'
                else:
                    prod_num = prod_map[(lhs, rhs)]
                    for a in follow_sets[lhs]:
                        if (i, a) in action_table:
                            # Conflict detection (shift/reduce or reduce/reduce)
                            action_table[(i, a)] += f'/reduce {prod_num}'
                        else:
                            action_table[(i, a)] = f'reduce {prod_num}'
        # GOTO for nonterminals
        for A in nonterminals:
            if (i, A) in goto_table:
                goto_table_out[(i, A)] = goto_table[(i, A)]
    print("=== ACTION TABLE ===")
    for k, v in action_table.items():
        print(f"State {k[0]}, Symbol '{k[1]}': {v}")
    print("FOLLOW(L):", follow_sets['L'])
    return action_table, goto_table_out