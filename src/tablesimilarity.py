import numpy as np
from statistics import mode
from scipy.optimize import linear_sum_assignment
import pandas as pd
import string
import random
import copy
import logging as log

def table_to_hash(table):
    """
    Creates hash entry for table meaning it returns a dictionary
    Every data element is represented as string
    Each unique cell element corresponds to a key in the dictionary
    The value of each dictionary for a key are x,y pers
    where the content was observed with x,y not in absolute counts
    but as x/n, y/m floating point representations

    Params:
    tables (list): A list of list corresponding to a table
        with tables[i,j] accessing row i and column j
    """
    m = len(table)
    if m == 0:
        return {}
    n = len(table[0])
    if n == 0:
        return {}
    result = {}
    for i in range(len(table)):
        for j in range(len(table[i])):
            key = str(table[i][j])
            vals = []
            if key in result:
                vals = result[key]
            vals.append([i / m, j / n])
            result[key] = vals
    return result


def test_table_to_hash():
    table = [[1, 2], [3, 3]]
    result = table_to_hash(table)
    assert isinstance(result, dict)
    assert len(result) == 3
    assert "1" in result
    assert "2" in result
    assert "3" in result
    assert not "4" in result
    assert result["1"] == [[0, 0]]
    assert result["2"] == [[0, 0.5]]
    assert result["3"] == [[0.5, 0], [0.5, 0.5]]


def point_dist(v1, v2):
    """
    Distance between two sets of points is approximated
    by distance bewteen means and standard deviations of points
    """
    a1 = np.asarray(v1)  # m rows and 2 columns
    a2 = np.asarray(v2)
    m1, n1 = a1.shape
    m2, n2 = a2.shape
    assert n1 == 2
    assert n2 == 2
    assert m1 > 0
    assert m2 > 0
    means1 = np.mean(a1, axis=0)  # column means = center of graviy
    means2 = np.mean(a2, axis=0)
    std1 = np.asarray([0, 0])
    std2 = np.asarray([0, 0])
    if n1 > 0:
        std1 = np.std(a1, axis=0)
    if n2 > 0:
        std2 = np.std(a2, axis=0)
    dm = dist = np.linalg.norm(means1 - means2)
    ds = dist = np.linalg.norm(std1 - std2)
    return dm + ds
  
  
def test_point_dist():
    v1 = [[1, 0], [2, 0], [3, 0]]
    v2 = [[2, 0], [3, 0], [4, 0]]
    v3 = [[1, 0], [3, 0], [5, 0]]
    result = point_dist(v1, v2)
    assert result == 1.0
    result2 = point_dist(v1, v3)
    assert result2 > 1.0


def table_hash_similarity(h1, h2, toplevel=True):
    """
    Computes similarity measure (value bewtween 0 and 1) between two tables.
    Parameters are hash representations of 2 tables as computed by
    `table_to_hash`

    Params:
    h1(dict): A hash representation of table 1 as computed by `table_to_hash`
    h2(dict): A hash representation of table 2 as computed by `table_to_hash`

    Returns (float): 1 for identical tables, 0 for tables with no elements in common
    """
    if not isinstance(h1, dict):
        return 0
    if not isinstance(h2, dict):
        return 0
    if len(h1) == 0 or len(h2) == 0:
        return 0
    if toplevel:  # return symmetrized result
        return 0.5 * (
            table_hash_similarity(h1, h2, toplevel=False)
            + table_hash_similarity(h2, h1, toplevel=False)
        )
    dist = 0
    keys1 = h1.keys()
    keys2 = h2.keys()
    found = 0
    for key in keys1:
        points1 = h1[key]
        if key in keys2:
            points2 = h2[key]
            dist = point_dist(points1, points2)
            assert dist >= 0
            # following logic ok bc point coordinates are in [0,1]
            sim = 1 - dist
            if sim < 0:
                sim = 0
            assert sim >= 0 and sim <= 1.0
            found += (
                sim  # found increemented by 1 if perfect match, smaller value otherwise
            )
    result = found / len(keys1)
    assert result >= 0 and result <= 1.0
    return result


def test_table_hash_similarity():
    t1 = [[1, 2], [3, 4]]
    t2 = [[1, 1], [3, 4]]
    h1 = table_to_hash(t1)
    h2 = table_to_hash(t2)
    assert table_hash_similarity(h1, h1) == 1.0
    assert table_hash_similarity(h2, h2) == 1.0
    result = table_hash_similarity(h1, h2)
    assert result > 0 and result < 1


def table_similarity(tables):
    """
    Table = list of list with same row and column lengths (but no need for uniform data type in colymns as in dataframes)
    For n tables, returns nxx similarity matrix.
    """
    n = len(tables)
    hashes = []
    result = np.zeros([n, n])
    if n == 0:
        return result
    for i in range(n):
        hashes.append(table_to_hash(tables[i]))
    for i in range(n):
        for j in range(i, n):
            result[i, j] = table_hash_similarity(hashes[i], hashes[j])
            result[j, i] = result[i, j]
    return result


def test_table_similarity():
    t1 = [["A", "B", "C"], [1, 2, 3]]
    t2 = [["A", "B", "D"], [7, 2, 3]]
    t3 = [["A", "D"], [7, 3]]
    tables = [t1, t2]
    result = table_similarity(tables)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    result2 = table_similarity([t1, t2, t3])
    print(result2)
    assert isinstance(result2, np.ndarray)
    assert result2.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            if i == j:
                assert result2[i, j] == 1.0
            else:
                assert result2[i, j] < 1.0


def compute_table_row_alignment(t1, t2, badcost=99):
    """
    # linear_sum_assignment(cost)
    """
    n = len(t1)
    n2 = len(t2)
    nm = max(len(t1), len(t2))
    # compute cost of assigning row i from table 1 to row j of table 2:
    costs = np.zeros([nm, nm])  # [nm,nm])
    costs.fill(badcost)
    for i in range(n):
        for j in range(n2):  # nm):
            assert i < len(t1)
            tr1 = [t1[i]]
            tr2 = [t2[j]]
            # compute similarity between two rows
            sims = table_similarity([tr1, tr2])
            cost = 1 - sims[0, 1]
            costs[i, j] = cost
    assignment = linear_sum_assignment(costs)[1]
    assignment = assignment.astype(float)
    assignment = assignment[0:n]
    for j in range(n):
        if assignment[j] >= n2:
            assignment[j] = np.nan
    return assignment


def test_compute_table_row_alignment():
    t1 = [[1, 2], [5, 6]]
    t2 = [[1, 2], [3, 4], [5, 6]]
    badcost = 99
    result = compute_table_row_alignment(t1, t2, badcost=99)
    assert len(result) == 2
    assert result[0] == 0
    assert result[1] == 2


def test_compute_table_row_alignment2():
    t1 = [["A", "B", "C", "D"], [1, 2, 3, 4], ["w", "x", "y", "z"]]
    t2 = [["A", "B", "D"], [1, 2, 4]]
    badcost = 99
    result = compute_table_row_alignment(t1, t2, badcost=99)
    assert len(result) == 3
    assert result[0] == 0
    assert result[1] == 1
    assert pd.isna(result[2])


def get_table_column(tbl, n, missing=np.nan):
    """
    Return column n of list of list
    """
    result = []
    for i in range(len(tbl)):
        if n >= len(tbl[i]):
            result.append(missing)
        else:
            result.append(tbl[i][n])
    return result


def compute_table_col_alignment(t1, t2, badcost=99, direction=1):
    """
    # linear_sum_assignment(cost)
    computes all to all column similarity
    """
    n = len(t1[0])
    n2 = len(t2[0])
    nm = max(n, n2)
    # compute cost of assigning row i from table 1 to row j of table 2:
    costs = np.zeros([nm, nm])  # n,n2])
    costs.fill(badcost)
    for i in range(n):
        for j in range(n2):
            tr1 = [get_table_column(t1, i)]
            tr2 = [get_table_column(t2, j)]
            # compute similarity between two rows
            sims = table_similarity([tr1, tr2])
            cost = 1 - sims[0, 1]
            costs[i, j] = cost
    assignment = linear_sum_assignment(costs)[direction]
    assignment = assignment.astype(float).tolist()
    for i in range(len(assignment)):
        if assignment[i] >= n2:
            assignment[i] = np.nan  # cannot be mapped to ary 2
    if len(assignment) > n:
        assignment = assignment[0:n]
    assert len(assignment) == n
    return assignment


def test_compute_table_col_alignment():
    t1 = [[1, 3], [4, 6], [7, 9]]
    t2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    badcost = 99
    result = compute_table_col_alignment(t1, t2, badcost=99)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == 0
    assert result[1] == 2


def test_compute_table_col_alignment2():
    t1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    t2 = [[1, 3], [4, 6], [7, 9]]
    badcost = 99
    result = compute_table_col_alignment(t1, t2, badcost=99)
    assert len(result) == 3
    assert result[0] == 0
    assert np.isnan(result[1])
    assert result[2] == 1


def compute_consensus_table_pair_offsets(t1, t2):
    """
    Finds offsets bewteen table indices.
    This is most difficult part because we are
    trying to find combinations of index offfsets
    which minimize gaps etc
    """
    m = len(t1)
    n = len(t1[0])
    m2 = len(t2)
    n2 = len(t2[0])
    result = np.zeros([m, n, 2], dtype=float)
    result.fill(np.nan)
    rowali = compute_table_row_alignment(t1, t2, badcost=99)
    colali = compute_table_col_alignment(t1, t2, badcost=99)
    for i in range(m):
        for j in range(n):
            assert i < len(rowali)
            assert j < len(colali)
            assert i < result.shape[0]
            assert j < result.shape[1]
            assert 1 < result.shape[2]
            if np.isnan(rowali[i]):
                continue
            result[i, j, 0] = rowali[i] - i
            assert (result[i, j, 0] + i) < len(t2)
            result[i, j, 1] = colali[j] - j
            # offsets can exceed boundary! Careful
            # assert (result[i,j,1]+j) < len(t2[0])
    return result


def test_compute_consensus_table_pair_offsets():
    t1 = [["A", "C"], [1, 3]]  # has one column missing
    t2 = [["A", "B", "C"], [1, 2, 3]]
    tall = [t1, t2]
    result = compute_consensus_table_pair_offsets(t1, t2)
    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 3
    nr = len(t1)
    m2 = len(t2)
    nc = len(t1[0])
    n2 = len(t2[0])
    for i in range(nr):
        for j in range(nc):
            origval = t1[i][j]
            if np.isnan(result[i, j, 0]):
                continue
            if np.isnan(result[i, j, 1]):
                continue
            rowo = int(result[i, j, 0])
            colo = int(result[i, j, 1])
            nappedval = np.nan
            if (i + rowo) < m2 and (j + colo) < n2:
                mappedval = t2[i + rowo][j + colo]
                assert mappedval == origval

    assert result[0, 0, 0] == 0  # row 0 aligns
    assert result[0, 0, 1] == 0  # col 0 aligns
    assert result[0, 1, 0] == 0  # row 1 aligns
    assert result[0, 1, 1] == 1  # col 1 aligns with col2


def test_compute_consensus_table_pair_offsets2():
    t1 = [["A", "B", "C", "D"], [1, 2, 3, 4]]
    t2 = [["A", "B", "D"], [1, 2, 4]]  # has one column missing
    tall = [t1, t2]
    result = compute_consensus_table_pair_offsets(t1, t2)
    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 3
    nr = len(t1)
    nc = len(t1[0])
    m2 = len(t2)
    n2 = len(t2[0])
    for i in range(nr):
        for j in range(nc):
            origval = t1[i][j]
            if np.isnan(result[i, j, 0]) or np.isnan(result[i, j, 1]):
                continue
            rowo = int(result[i, j, 0])
            colo = int(result[i, j, 1])
            mappedval = np.nan
            if (i + rowo) < m2 and (j + colo) < n2:
                mappedval = t2[i + rowo][j + colo]
                # print(f"row:{i},col:{j} off: {rowo},{colo} : mapped val:", mappedval)
                assert mappedval == origval
    # assert result[0,0,0] == 0 # row 0 aligns
    # assert result[0,0,1] == 0 # col 0 aligns
    # assert result[0,1,0] == 0 # row 1 aligns
    # assert result[0,1,1] == 1  # col 1 aligns with col2


def test_compute_consensus_table_pair_offsets3():
    t1 = [["A", "B", "C", "D"], [1, 2, 3, 4], ["w", "x", "y", "z"]]
    t2 = [["A", "B", "D"], [1, 2, 4]]  # has one column missing

    tall = [t1, t2]
    result = compute_consensus_table_pair_offsets(t1, t2)
    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 3
    nr = len(t1)
    nc = len(t1[0])
    m2 = len(t2)
    n2 = len(t2[0])
    for i in range(nr):
        for j in range(nc):
            origval = t1[i][j]
            if np.isnan(result[i, j, 0]) or np.isnan(result[i, j, 1]):
                continue
            rowo = int(result[i, j, 0])
            colo = int(result[i, j, 1])
            # print("orig i,j:",i,j,origval,'new',i+rowo,j+colo)
            mappedval = np.nan
            if (i + rowo) < m2 and (j + colo) < n2:
                mappedval = t2[i + rowo][j + colo]
                # print(f"row:{i},col:{j} off: {rowo},{colo} : mapped val:", mappedval)
                assert mappedval == origval
    # assert result[0,0,0] == 0 # row 0 aligns
    # assert result[0,0,1] == 0 # col 0 aligns
    # assert result[0,1,0] == 0 # row 1 aligns
    # assert result[0,1,1] == 1  # col 1 aligns with col2


def compute_consensus_table_offsets(tables):
    """
    Finds offsets bewteen table indices.
    This is most difficult part because we are
    trying to find combinations of index offfsets
    which minimize gaps etc
    """
    tbl = tables[0]
    m = len(tbl)
    n = len(tbl[0])
    k = len(tables)
    # although we computer integer offsets
    # we need float so that we can represent NaN:
    result = np.zeros([m, n, 2, k - 1], dtype=float)
    if k < 2:
        return result
    # TODO
    assert len(result.shape) == 4
    for t in range(1, k):
        off = compute_consensus_table_pair_offsets(tables[0], tables[t])
        assert isinstance(off, np.ndarray)
        for i in range(m):
            for j in range(n):
                for d in range(2):
                    # print("accessing",i,j,d,t-1)
                    result[i, j, d, t - 1] = off[i, j, d]
    return result


def test_compute_consensus_table_offsets():
    t1 = [["A", "C"], [1, 3]]
    # t2 = [['A','B','D'],[7,2,3]]
    t2 = [["A", "B", "C"], [1, 2, 3]]
    tall = [t1, t2]
    result = compute_consensus_table_offsets(tall)
    assert len(result.shape) == 4
    nr, nc, d, nt = result.shape
    for i in range(nr):
        for j in range(nc):
            origval = t1[i][j]
            for t in range(nt):
                if np.isnan(result[i, j, 0, t]) or np.isnan(result[i, j, 1, t]):
                    continue
                rowo = int(result[i, j, 0, t])
                colo = int(result[i, j, 1, t])
                assert i + rowo < len(tall[t + 1])
                assert j + colo < len(tall[t + 1][0])
                mappedval = tall[t + 1][i + rowo][j + colo]
                # print(f"row:´{i},col:{j} table: {t+1} off: {rowo},{colo} : mapped val:", mappedval)
                assert mappedval == origval
    assert result[0, 0, 0, 0] == 0
    assert result[0, 0, 0, 0] == 0
    assert result[0, 1, 0, 0] == 0
    assert result[0, 1, 1, 0] == 1  # mappting 'C' ot t3 to t1


def list_dimensions(lst):
    result = []
    if isinstance(lst, list):
        result.append(len(lst))
        if len(lst) > 0:
            result.extend(list_dimensions(lst[0]))
    return result


def test_list_dimensions():
    lst = [[1, 2, 3], [4, 5, 6]]
    dims = list_dimensions(lst)
    assert len(dims) == 2
    assert dims == [2, 3]


def compute_consensus_table(tables):
    """
    Computes consensus between set of related tables
    """
    if len(tables) == 0:
        return None
    reftbl = tables[0]
    if len(tables) == 1:
        return reftbl
    offsets = compute_consensus_table_offsets(tables)
    # result = reftbl.copy() # .copy is not really deep copy only deep copy of first layer not recursively
    result = copy.deepcopy(reftbl)
    m, n, o, t = offsets.shape  # t: number of tables
    for i in range(m):  # rows
        for j in range(n):  # columns
            vals = [reftbl[i][j]]
            for k in range(t):  # tables
                o1 = offsets[i, j, 0, k]
                o2 = offsets[i, j, 1, k]
                if np.isnan(o1) or np.isnan(o2):
                    continue
                row = i + int(o1)
                col = j + int(o2)
                if row < len(tables[k + 1]) and col < len(tables[k + 1][row]):
                    vals.append(tables[k + 1][row][col])
                else:
                    print(
                        "warning: inconsistent index",
                        row,
                        col,
                        len(tables[k + 1]),
                        len(tables[k + 1][row]),
                    )
                    print(tables[k + 1])
            result[i][j] = mode(vals)  # replace with most common value
    return result


def test_compute_consensus_table():
    t1 = [["A", "B", "C"], [1, 2, 3]]
    t2 = [["A", "B", "D"], [7, 2, 3]]
    t3 = [["A", "D"], [7, 3]]
    result = compute_consensus_table([t1, t2, t3])
    assert len(result) == len(t1)
    assert len(result[0]) == len(t1[0])
    assert result == [
        ["A", "B", "D"],
        [7, 2, 3],
    ]  # 'D' and '7' are changed due to consensus
    t4 = [["A", "B"], ["C", "D"]]
    t5 = [["A", "B"], ["C", "d"]]
    result = compute_consensus_table([t4, t5])  # [t1,t2,t3])
    assert result == t4
    result = compute_consensus_table([t4, t5, t5])  # [t1,t2,t3])
    assert result == t5
    t6 = [["A"]]
    t7 = [["B"]]
    result = compute_consensus_table([t6, t7, t7])  # [t1,t2,t3])
    assert result == t7


def test_compute_consensus_table2():
    t1 = [["A", "B", "C"], [1, 2, 3]]
    t2 = [["A"], [7]]
    t3 = [["A", "D"], [7, 3]]
    result = compute_consensus_table([t1, t2, t3])
    assert len(result) == len(t1)
    assert len(result[0]) == len(t1[0])
    assert result == [["A", "B", "C"], [7, 2, 3]]  # '7' changed due to consensus


def table_assignment(tlist1, tlist2, badval=99, score_ths=0.50):
    """
    Given a list of list of tables (corresponding to table extractions
    from different AI services), this function finds the
    best correspondence with respect to the first list of tables.
    This is important in case the different AI services
    have table extractions that are not necessarilcy in same
    order or even same count of tables. It does
    assume that the first list of extracted tables is
    'leading' in the sense that it defines to amount and order
    of consensus table computations to perform.
    """
    # print("starting table assignment with", tlist1,'and', tlist2)
    if len(tlist1) == 0:
        # print("returning empty assignment because first list was empty")
        return []
    assert len(tlist1) > 0  # first list must have at lest one table
    assert len(tlist1[0]) > 0  # first table of first list must have at least one row
    assert (
        len(tlist1[0][0]) > 0
    )  # first row of first table of first list must have at least one column
    assert not isinstance(
        tlist1[0][0][0], list
    )  # must be table cell element not further high dimensional lists
    m = len(tlist1)
    n = len(tlist2)
    mm = max(m, n)
    row = [badval] * mm
    result = []
    for i in range(mm):
        result.append(copy.deepcopy(row))
    for i in range(m):
        lst = []
        h1 = table_to_hash(tlist1[i])
        for j in range(n):
            h2 = table_to_hash(tlist2[j])
            sim = table_hash_similarity(h1, h2)
            dist = 1 - sim
            result[i][j] = dist
    assignment = linear_sum_assignment(result)[0]
    assignment = assignment.astype(float)
    for i in range(len(assignment)):
        if result[i][int(assignment[i])] > score_ths:
            assignment[i] = np.nan
    if isinstance(assignment, np.ndarray):
        assignment = assignment.tolist()
    assert isinstance(assignment, list)
    if len(assignment) > len(tlist1):
        assignment = assignment[: len(tlist1)]
    if len(assignment) != len(tlist1):
        print("strange assignment:", len(assignment), len(tlist1))
        print(assignment)
        print(tlist1)
    assert len(assignment) == len(tlist1)
    return assignment


def test_table_assignment():
    t1a = [[1, 2], [3, 4]]
    t1b = [[1, 2], [3, 4], [5, 6]]
    t1c = [[1, 2], [3, 44]]
    t2a = [["A", "B"], ["C", "D"]]
    t2b = [["A", "B"], ["C", "d"]]
    t2c = [["A", "Z", "B"], ["C", "z", "D"]]
    t3a = [["a", "b"], ["c", "d"]]
    t3b = [["z", "a", "b"], ["z", "c", "d"]]
    series1 = [t1a, t2a, t3a]
    series2 = [t1b, t2b]
    series3 = [t1c, t3b]
    result1 = table_assignment(series1, series2)
    assert len(result1) == len(series1)
    assert result1[0] == 0
    assert result1[1] == 1
    assert np.isnan(result1[2])


def organize_table_comparisons(tlists):
    """
    Organize lists of tables obtained from multiple AI sources
    such that most similar tables from each list are assigned
    so that a consensus approach is launched for each
    'cluster' of similar tables

    Params:
    tlists(list): A list on m AI services that returned a variable number
    of tables. The first list is assumed to be of highest quality
    and used as reference list. For examples, if Tetraxt, Google Vision and Azure
    returned for a particular submitted page 2, 3 and 4 tables respetively,
    the variable tlists is a list with 3 elements, each of them are again
    lists of length 2,3 and 1 respectively. The resulting list has length 2
    (because the first listed service detected 2 tables). The first list lement is a list
    of 3 tables: firstly the first table tom the first service, then the most similar
    table from the second service and thirdly the most similar table from the third service
    The second list element of this example results is again a list, with the first
    element being the list return second by the first service, followed by the
    most similar table from the second service and the most similar service from the third service.
    """
    assignments = []
    nservices = len(tlists)
    for i in range(1, nservices):
        lst = table_assignment(tlists[0], tlists[i])
        if len(lst) != len(tlists[0]):
            print("inconsistent assignment:")
            print(tlists[0])
            print(tlists[i])
            print(lst)
        assert len(lst) == len(
            tlists[0]
        )  # log.warning("inconsistent number of table assignments")

        assignments.append(lst)
    assert len(assignments) == (len(tlists) - 1)
    result = []
    # loop over tables from first series (= first AI ervice)
    # for each table we try to sind equivalnt tables
    # from other AI services so that we can compute consensus
    nreftables = len(tlists[0])
    for i in range(nreftables):
        lst = [tlists[0][i]]
        # loop over results form other table extraction ionresults::
        for j in range(len(assignments)):
            assert len(assignments[j]) == nreftables
            index = assignments[j][i]
            if not np.isnan(index):
                assert index < len(tlists[j + 1])
                assert not np.isnan(index)
                lst.append(tlists[j + 1][int(index)])
        result.append(lst)
    assert len(result) == len(tlists[0])
    return result


def compute_consensus_tables(tablelists):
    """
    Central function: given a list of list of tables
    corresponding to table extractions from different AI
    services, compute consensus table results for each
    table with respect to the first sub-list of tables
    """
    if len(tablelists) == 0:
        return []
    if len(tablelists) < 3:  # cannot compute consensus if only 1 or 2 possibilities
        return tablelists[0]
    orig = copy.deepcopy(tablelists)
    if not isinstance(tablelists, list):
        log.error(
            "internal error: called computer_consensus_tables with incorrect data type"
        )
        return []
    if len(tablelists) == 0:
        log.error("internal error: called compute_consensus_tables with empty input")
        return []
    nservices = len(tablelists)
    nreftables = len(tablelists[0])
    organized_tables = organize_table_comparisons(tablelists)
    assert len(organized_tables) == nreftables
    result = []
    for i in range(len(organized_tables)):
        lst = organized_tables[i]
        # assert len(other_ids) == (nservices-1) # number of ids from ohter AI services
        # for j in range(len(other_ids)):
        #     k = other_ids[j]
        #     if np.isnan(k):
        #         continue
        #     else:
        #         assert (j+1) < len(tablelists)
        #         assert k < len(tablelists[j+1])
        #         lst.append(tablelists[j+1][k])
        consensus_table = compute_consensus_table(lst)
        result.append(consensus_table)
    assert len(result) == nreftables
    assert orig == tablelists
    return result


def test_compute_consensus_tables():
    t1a = [[1, 2], [3, 4]]
    t1b = [[1, 2], [3, 4], [5, 6]]
    t1c = [[1, 2], [3, 44]]
    t2a = [["A", "B"], ["C", "D"]]
    t2b = [["A", "B"], ["C", "d"]]
    t2c = [["A", "Z", "B"], ["C", "z", "D"]]
    t3a = [["a", "b"], ["c", "d"]]
    t3b = [["z", "a", "b"], ["z", "c", "d"]]
    series1 = [t1a, t2a, t3a]
    series2 = [t1b, t2b]
    series3 = [t1c, t3b]
    result1 = compute_consensus_tables([series1, series2, series3])
    assert len(result1) == len(series1)


def compute_consensus_dataframes(dflists: "list[list[pd.DataFrame]]") -> "list[pd.DataFrame]":
    """
    Central function: given a list of list of dataframes
    corresponding to table extractions from different AI
    services, compute consensus table results for each
    table with respect to the first sub-list of tables
    """
    tablelists = []
    # convert dataframes to lists o f lsits:
    for i in range(len(dflists)):
        lst = []
        for j in range(len(dflists[i])):
            lst.append(dflists[i][j].values.tolist())
        tablelists.append(lst)
    result = compute_consensus_tables(tablelists)
    # convert list of lists back to dataframes:
    result2 = []
    for i in range(len(result)):
        if i < len(dflists[0]):  # col names takesn from first service:
            result2.append(pd.DataFrame(result[i], columns=dflists[0][i].columns))
    return result2


def generate_random_string(nchars, alphabet=(string.ascii_uppercase + string.digits)):
    """
    Returns string with length nchars consisting of letters of provided alphabet
    """
    return "".join(random.choices(alphabet, k=nchars))


def test_generate_random_string():
    for i in range(5):
        s = generate_random_string(i)
        assert len(s) == i
        assert isinstance(s, str)


def generate_random_table_row(
    nchars, nwords, alphabet=(string.ascii_uppercase + string.digits)
):
    """
    This internal function generates a list with nwords items
    filled with randomized strings of length nchar given by the provided alphabet.
    """
    result = []
    for i in range(nwords):
        result.append(generate_random_string(nchars, alphabet=alphabet))
    return result


def generate_random_table(
    nrows, ncols, nchars=6, alphabet=(string.ascii_uppercase + string.digits)
):
    """
    This function generates a list of list with nrows rows and ncols columns,
    filled with randomized strings given by the provided alphabet.
    """
    result = []
    for i in range(nrows):
        result.append(
            generate_random_table_row(nchars=nchars, nwords=ncols, alphabet=alphabet)
        )
    return result


def test_generate_random_table():
    a = generate_random_table(nrows=3, ncols=4, nchars=6)
    assert len(a) == 3
    assert len(a[0]) == 4
    assert len(a[0][0]) == 6


def test_compute_consensus_tables_stresstest(itermax=100):
    for iteration in range(itermax):
        nservices = random.randint(1, 5)
        tablelists = []
        for service in range(nservices):
            servicetables = []
            ntabels = random.randint(0, 5)
            for i in range(ntabels):
                nrows = random.randint(1, 5)
                ncols = random.randint(1, 5)
                nchars = random.randint(0, 5)
                table = generate_random_table(nrows=nrows, ncols=ncols, nchars=nchars)
                servicetables.append(table)
            tablelists.append(servicetables)
        # assert len(tablelists) > 0
        # assert len(tablelists[0]) > 0
        # print("iteration", iteration)
        # print(tablelists)
        result = compute_consensus_tables(tablelists)
        # print('result:', result)
        assert isinstance(result, list)
        assert len(result) == len(tablelists[0])


def test_compute_consensus_tables_stresstest2(itermax=100):
    for iteration in range(itermax):
        # print("iteration", iteration)
        nservices = random.randint(1, 5)
        tablelists = []
        for j in range(nservices):
            tablelists.append([])
        servicetables = []
        ntabels = random.randint(1, 5)
        for i in range(ntabels):
            nrows = random.randint(1, 5)
            ncols = random.randint(1, 5)
            nchars = random.randint(0, 5)
            table = generate_random_table(nrows=nrows, ncols=ncols, nchars=nchars)
            for j in range(nservices):
                tablelists[j].append(copy.deepcopy(table))
            # servicetables.append(table)
        # servicetables_orig = servicetables.copy()
        # print('tablelists:')
        # print(tablelists)
        # print('First table of original first list:')
        # print(tablelists[0][0])
        correct_word = tablelists[0][0][0][0]
        # print('original first word:', correct_word)
        assert isinstance(correct_word, str)
        typo_word = "DUMMY_WORD"
        # for i in range(nservices-1):
        #    servicetables2 = servicetables_orig.copy()
        #    tablelists.append(servicetables2)
        # tablelists.append(servicetables)
        assert tablelists[0][0][0][0] != typo_word
        assert nservices < 2 or tablelists[1][0][0][0] != typo_word
        tablelists[0][0][0][0] = typo_word  # change and check later if corrected
        # print('changed first word:', tablelists[0][0][0][0])
        # print("submitted first words:")
        # for i in range(len(tablelists)):
        #    # print(tablelists[i][0][0][0])
        assert tablelists[0][0][0][0] == typo_word
        assert nservices < 2 or tablelists[1][0][0][0] != typo_word
        result = compute_consensus_tables(tablelists)
        # print('result:', result)
        assert isinstance(result, list)
        assert len(result) == len(tablelists[0])
        # print("submitted first words2:")
        # for i in range(len(tablelists)):
        #    print(tablelists[i][0][0][0])
        assert tablelists[0][0][0][0] == typo_word
        # assert len(tablelists) > 0
        # assert len(tablelists[0]) > 0
        # print("iteration", iteration)
        # print(tablelists)
        result = compute_consensus_tables(tablelists)
        # print('result:', result)
        assert isinstance(result, list)
        assert len(result) == len(tablelists[0])
        # print("this is original first table:")
        # print(tablelists[0][0])
        # print("This is corrected first table:")
        # print(result[0])
        # correction only required if not strange table with one row and one column::
        # correction only required if 3 or more services:
        assert (
            (nservices < 3)
            or (len(result[0]) < 2 or (len(result[0][0]) < 2))
            or result[0][0][0] == correct_word
        )  # see if consensus corrected modified word


def test_compute_consensus_dataframes_stresstest2(itermax=100):
    for iteration in range(itermax):
        # print("iteration", iteration)
        nservices = random.randint(1, 5)
        tablelists = []
        for j in range(nservices):
            tablelists.append([])
        servicetables = []
        ntabels = random.randint(1, 5)
        for i in range(ntabels):
            nrows = random.randint(1, 5)
            ncols = random.randint(1, 5)
            nchars = random.randint(0, 5)
            table = generate_random_table(nrows=nrows, ncols=ncols, nchars=nchars)
            colnames = generate_random_table_row(nwords=ncols, nchars=10)
            df = pd.DataFrame(table, columns=colnames)
            for j in range(nservices):
                tablelists[j].append(copy.deepcopy(df))
        print("First table of first service:", tablelists[0][0])
        # change first work of first table from first service:
        correct_word = tablelists[0][0].iloc[0][0]
        # print('original first word:', correct_word)
        assert isinstance(correct_word, str)
        typo_word = "DUMMY_WORD"
        assert tablelists[0][0].iloc[0, 0] != typo_word
        assert nservices < 2 or tablelists[1][0].iloc[0, 0] != typo_word
        tablelists[0][0].iloc[0, 0] = typo_word  # change and check later if corrected
        # print('changed first word:', tablelists[0][0])
        assert tablelists[0][0].iloc[0, 0] == typo_word
        assert nservices < 2 or tablelists[1][0].iloc[0, 0] != typo_word
        result = compute_consensus_dataframes(tablelists)
        # print('result:', result)
        assert isinstance(result, list)
        assert isinstance(result[0], pd.DataFrame)
        assert len(result) == len(tablelists[0])
        # print("submitted first words2:")
        # for i in range(len(tablelists)):
        #    print(tablelists[i][0][0][0])
        assert tablelists[0][0].iloc[0, 0] == typo_word
        # assert len(tablelists) > 0
        # assert len(tablelists[0]) > 0
        # print("iteration", iteration)
        # print(tablelists)
        # print('result:', result)
        # correction only required if not strange table with one row and one column::
        # correction only required if 3 or more services:
        assert (
            (nservices < 3)
            or (len(result[0]) < 2 or (len(result[0].columns) < 2))
            or result[0].iloc[0, 0] == correct_word
        )  # see if consensus corrected modified word
        print("original first table:", tablelists[0][0])
        print("fixed:", result[0])
        assert list(result[0].columns) == list(
            tablelists[0][0].columns
        )  # column names must be preserverd


def test_all():
    test_compute_consensus_dataframes_stresstest2()
    test_compute_consensus_table()
    test_compute_consensus_tables_stresstest()
    test_compute_consensus_tables_stresstest2()
    test_generate_random_string()
    test_generate_random_table()
    test_list_dimensions()
    test_compute_consensus_tables()
    test_table_assignment()
    test_compute_consensus_table()
    test_compute_consensus_table2()
    test_compute_table_row_alignment()
    test_compute_table_row_alignment2()
    test_compute_consensus_table_pair_offsets3()
    test_compute_table_col_alignment()
    test_compute_table_col_alignment2()
    test_compute_consensus_table_pair_offsets2()
    test_compute_consensus_table_pair_offsets()
    test_compute_consensus_table_offsets()
    test_compute_table_row_alignment()
    test_table_similarity()
    test_table_to_hash()
    test_table_hash_similarity()
    test_point_dist()


if __name__ == "__main__":
    test_all()
