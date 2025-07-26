query_val = None
def get_query():
    global query
    q = input("Query: ") + " site:github.com"
    query_val = q
    return q