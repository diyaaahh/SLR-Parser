import streamlit as st
from parser_utils import parse_grammar, compute_first_sets, compute_follow_sets, construct_lr0_item_sets, goto, build_slr_parsing_table
import pandas as pd

# Fixed grammar and input string from the question
grammar_text = """S' -> S\nS -> L = R\nS -> R\nL -> * R\nL -> id\nR -> L"""
input_string = "id = id * id"

st.set_page_config(page_title="SLR Parser Visualizer", layout="wide")
st.title("SLR Parser Visualizer")

# Display grammar and input string (read-only)
st.markdown("""
**Grammar:**
```
S' -> S
S -> L = R
S -> R
L -> * R
L -> id
R -> L
```
**Input String:**
```
id = id * id
```
""")

# Parse grammar and compute FIRST/FOLLOW sets
grammar = parse_grammar(grammar_text)
first_sets = compute_first_sets(grammar)
follow_sets = compute_follow_sets(grammar, first_sets, start_symbol="S'")

# Compute LR(0) item sets and GOTO table
item_sets, goto_table = construct_lr0_item_sets(grammar)

# Build a mapping from frozenset(item_set) to index for fast lookup
item_set_map = {frozenset(item_set): idx for idx, item_set in enumerate(item_sets)}

# Gather all grammar symbols
symbols = set()
for lhs, rhs in grammar:
    symbols.update(rhs)
    symbols.add(lhs)
symbols = sorted(symbols)

# Build SLR parsing table
action_table, goto_table_out = build_slr_parsing_table(grammar, item_sets, goto_table, follow_sets)

# Get terminals and non-terminals
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
terminals = sorted(terminals)
nonterminals = sorted(nonterminals)
nonterminals = sorted(nt for nt in nonterminals if nt != "S'")

def item_to_str(lhs, rhs, dot_pos):
    rhs_with_dot = list(rhs)
    rhs_with_dot.insert(dot_pos, '•')
    return f"{lhs} -> {' '.join(rhs_with_dot)}"

# Create a mapping to track how each item set was created
def build_predecessor_map(item_sets, goto_table):
    """Build a mapping from item set index to its predecessor information"""
    predecessor_map = {}
    predecessor_map[0] = None  # Initial state has no predecessor
    
    for (state_idx, symbol), target_idx in goto_table.items():
        if target_idx not in predecessor_map:
            predecessor_map[target_idx] = (state_idx, symbol)
    
    return predecessor_map

predecessor_map = build_predecessor_map(item_sets, goto_table)

# Main area with tabs
tabs = st.tabs(["FIRST & FOLLOW Sets", "SLR Item Sets", "Parsing Table", "Parsing Steps"])

with tabs[0]:
    st.subheader("FIRST & FOLLOW Sets")
    st.markdown("### FIRST Sets")
    # Exclude S' from FIRST sets display
    first_display = {nt: ', '.join(sorted(s)) for nt, s in first_sets.items() if nt != "S'"}
    st.table(first_display)
    st.markdown("### FOLLOW Sets")
    # Exclude S' from FOLLOW sets display
    follow_display = {nt: ', '.join(sorted(s)) for nt, s in follow_sets.items() if nt != "S'"}
    st.table(follow_display)

with tabs[1]:
    st.subheader("SLR Item Sets")
    for idx, item_set in enumerate(item_sets):
        # Create header with predecessor information
        if idx == 0:
            header = f"Item Set I{idx} (Initial State)"
        else:
            pred_info = predecessor_map.get(idx)
            if pred_info:
                pred_state, symbol = pred_info
                header = f"Item Set I{idx} = GOTO(I{pred_state}, {symbol})"
            else:
                header = f"Item Set I{idx}"
        
        with st.expander(header):
            st.markdown("**Items:**")
            for item in sorted(item_set):
                st.write(item_to_str(*item))

with tabs[2]:
    st.subheader("SLR Parsing Table")
    st.markdown("#### SLR Parsing Table")
    # Prepare columns for MultiIndex
    action_syms = terminals
    goto_syms = nonterminals
    n_states = len(item_sets)
    columns = (
        [("State", "State")] +
        [("Action", t) for t in action_syms] +
        [("Goto", nt) for nt in goto_syms]
    )
    data = []
    for state in range(n_states):
        row = [f"I{state}"]
        for t in action_syms:
            val = action_table.get((state, t), "")
            row.append(val)
        for nt in goto_syms:
            val = goto_table_out.get((state, nt), "")
            row.append(str(val) if val != "" else "")
        data.append(row)
    df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
    def highlight_conflicts(val):
        if isinstance(val, str) and 'shift' in val and 'reduce' in val:
            return 'background-color: #ffcccc; color: red; font-weight: bold'
        return ''
    styled_df = df.style.applymap(highlight_conflicts)
    st.dataframe(styled_df, use_container_width=True)

with tabs[3]:
    st.subheader("Parsing Steps")
    st.markdown("#### Step-by-step SLR Parsing for input: 'id = id * id'")
    # SLR parsing simulation
    input_tokens = ['id', '=', 'id', '*', 'id', '$']
    stack = [0]
    steps = []
    pointer = 0
    max_steps = 50
    while pointer < len(input_tokens) and len(steps) < max_steps:
        state = stack[-1]
        symbol = input_tokens[pointer]
        action = action_table.get((state, symbol), '')
        # Detect shift/reduce conflict
        conflict = 'shift' in action and 'reduce' in action
        steps.append({
            'Stack': ' '.join(map(str, stack)),
            'Input': ' '.join(input_tokens[pointer:]),
            'Action': action,
            'Conflict': 'Yes' if conflict else ''
        })
        if conflict:
            break
        if action.startswith('shift'):
            next_state = int(action.split()[1])
            stack.append(symbol)
            stack.append(next_state)
            pointer += 1
        elif action.startswith('reduce'):
            prod_num = int(action.split()[1])
            lhs, rhs = grammar[prod_num]
            if rhs != ['ε']:
                for _ in range(2 * len(rhs)):
                    stack.pop()
            state = stack[-1]
            stack.append(lhs)
            goto_state = goto_table_out.get((state, lhs), None)
            if goto_state is None:
                steps[-1]['Action'] += ' (Goto error)'
                break
            stack.append(goto_state)
        elif action == 'accept':
            steps[-1]['Conflict'] = ''
            break
        else:
            steps[-1]['Action'] += ' (Error)'
            break
    st.markdown("**Legend:** Red row = shift/reduce conflict encountered.")
    st.table(steps)