""""
modified version of: https://github.com/3b1b/videos/tree/master/_2022/wordle
"""


import numpy as np
import itertools as it

MISS = np.uint8(0)
MISPLACED = np.uint8(1)
EXACT = np.uint8(2)

# Generating color patterns between strings, etc.

def words_to_int_arrays(words):
    return np.array([[ord(c)for c in w] for w in words], dtype=np.uint8)


def generate_pattern_matrix(words1, words2):
    """
    A pattern for two words represents the wordle-similarity
    pattern (grey -> 0, yellow -> 1, green -> 2) but as an integer
    between 0 and 3^5. Reading this integer in ternary gives the
    associated pattern.

    This function computes the pairwise patterns between two lists
    of words, returning the result as a grid of hash values. Since
    this can be time-consuming, many operations that can be are vectorized
    (perhaps at the expense of easier readibility), and the the result
    is saved to file so that this only needs to be evaluated once, and
    all remaining pattern matching is a lookup.
    """

    # Number of letters/words
    nl = len(words1[0])
    nw1 = len(words1)  # Number of words
    nw2 = len(words2)  # Number of words

    # Convert word lists to integer arrays
    word_arr1, word_arr2 = map(words_to_int_arrays, (words1, words2))

    # equality_grid keeps track of all equalities between all pairs
    # of letters in words. Specifically, equality_grid[a, b, i, j]
    # is true when words[i][a] == words[b][j]
    equality_grid = np.zeros((nw1, nw2, nl, nl), dtype=bool)
    for i, j in it.product(range(nl), range(nl)):
        equality_grid[:, :, i, j] = np.equal.outer(word_arr1[:, i], word_arr2[:, j])

    # full_pattern_matrix[a, b] should represent the 5-color pattern
    # for guess a and answer b, with 0 -> grey, 1 -> yellow, 2 -> green
    full_pattern_matrix = np.zeros((nw1, nw2, nl), dtype=np.uint8)

    # Green pass
    for i in range(nl):
        #print("Green Pass: " + str(i/nl*100) + '%', end="\r")
        matches = equality_grid[:, :, i, i].flatten()  # matches[a, b] is true when words[a][i] = words[b][i]
        full_pattern_matrix[:, :, i].flat[matches] = EXACT

        for k in range(nl):
            # If it's a match, mark all elements associated with
            # that letter, both from the guess and answer, as covered.
            # That way, it won't trigger the yellow pass.
            equality_grid[:, :, k, i].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False

    # Yellow pass
    for i, j in it.product(range(nl), range(nl)):
        #print("Yellow Pass: " + str(i/nl*100) + '%', end="\r")
        matches = equality_grid[:, :, i, j].flatten()
        full_pattern_matrix[:, :, i].flat[matches] = MISPLACED
        for k in range(nl):
            # Similar to above, we want to mark this letter
            # as taken care of, both for answer and guess
            equality_grid[:, :, k, j].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False

    # Rather than representing a color pattern as a lists of integers,
    # store it as a single integer, whose ternary representations corresponds
    # to that list of integers.
    pattern_matrix = np.dot(
        full_pattern_matrix,
        (3**np.arange(nl)).astype(np.uint8)
    )

    return pattern_matrix


def get_pattern_matrix(words1, words2):
    PATTERN_GRID_DATA = dict()
 
    pattern_matrix = generate_pattern_matrix(words2, words2)
    PATTERN_GRID_DATA['grid'] = pattern_matrix
    PATTERN_GRID_DATA['words_to_index'] = dict(zip(
        words2, it.count()
    ))

    full_grid = PATTERN_GRID_DATA['grid']
    words_to_index = PATTERN_GRID_DATA['words_to_index']

    indices1 = [words_to_index[w] for w in words1]
    indices2 = [words_to_index[w] for w in words2]
    return full_grid[np.ix_(indices1, indices2)]


def get_pattern(guess, answer):
    return generate_pattern_matrix([guess], [answer])[0, 0]


def pattern_from_string(pattern_string):
    return sum((3**i) * int(c) for i, c in enumerate(pattern_string))


def pattern_to_int_list(pattern):
    result = []
    curr = pattern
    for x in range(5):
        result.append(curr % 3)
        curr = curr // 3
    return result


def pattern_to_string(pattern):
    d = {MISS: "⬛", MISPLACED: "🟨", EXACT: "🟩"}
    return "".join(d[x] for x in pattern_to_int_list(pattern))


def patterns_to_string(patterns):
    return "\n".join(map(pattern_to_string, patterns))


def get_possible_words(guess, pattern, word_list):
    all_patterns = get_pattern_matrix([guess], word_list).flatten()
    return list(np.array(word_list)[all_patterns == pattern])




# Functions associated with entropy calculation

def get_pattern_distributions(allowed_words, possible_words):
    """
    For each possible guess in allowed_words, this finds the probability
    distribution across all of the 3^5 wordle patterns you could see, assuming
    the possible answers are in possible_words with associated probabilities
    in weights.

    It considers the pattern hash grid between the two lists of words, and uses
    that to bucket together words from possible_words which would produce
    the same pattern, adding together their corresponding probabilities.
    """
    pattern_matrix = get_pattern_matrix(allowed_words, possible_words)

    n = len(allowed_words)
    n_p = len(possible_words)
    distributions = np.zeros((n, 3**5))
    n_range = np.arange(n)
    for j in range(n_p):
        distributions[n_range, pattern_matrix[:, j]] += 1
    return distributions


# Run simulated wordle games

def simulate_games(bot,
                   word_list,
                   quiet=False,
                   ):
    bot.reset()
    first_guess = bot.best_pick()


    # Go through each answer in the test set, play the game,
    # and keep track of the stats.
    scores = np.zeros(0, dtype=int)
    game_results = []
    for answer in word_list:
        bot.reset()
        guesses = []
        patterns = []
        possibility_counts = []
        possibilities = word_list.copy()

        score = 1
        guess = first_guess
        round_n=1
        while guess != answer and round_n <5:
            pattern = get_pattern(guess, answer)
            guesses.append(guess)
            patterns.append(pattern)
            possibilities = get_possible_words(guess, pattern, bot.allowed_word_list)
            possibility_counts.append(len(possibilities))
            bot.initialize_for_next_round(possibilities)

            score += 1
            round_n+=1
            guess = bot.best_pick()


        # Accumulate stats
        scores = np.append(scores, [score])
        score_dist = [
            int((scores == i).sum())
            for i in range(1, scores.max() + 1)
        ]
        total_guesses = scores.sum()
        average = scores.mean()

        game_results.append(dict(
            score=int(score),
            answer=answer,
            guesses=guesses,
            patterns=list(map(int, patterns)),
            reductions=possibility_counts,
        ))
        # Print outcome
        if not quiet:
            message = "\n".join([
                "",
                f"Score: {score}",
                f"Answer: {answer}",
                f"Guesses: {guesses}",
                f"Reductions: {possibility_counts}",
                *patterns_to_string((*patterns, 3**5 - 1)).split("\n"),
                *" " * (6 - len(patterns)),
                f"Distribution: {score_dist}",
                f"Total guesses: {total_guesses}",
                f"Average: {average}",
                *" " * 2,
            ])
            if answer is not word_list[0]:
                # Move cursor back up to the top of the message
                n = len(message.split("\n")) + 1
                print(("\033[F\033[K") * n)
            else:
                print("\r\033[K\n")
            print(message)

    final_result = dict(
        score_distribution=score_dist,
        total_guesses=int(total_guesses),
        average_score=float(scores.mean()),
        game_results=game_results,
    )

  
    return final_result
