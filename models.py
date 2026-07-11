import json
from database import get_db


def save_test_result(user_id: int, result_data: dict) -> int:
    conn   = get_db()
    cursor = conn.cursor()

    emotions = result_data.get("emotions", {})
    links    = result_data.get("links", {})

    cursor.execute(
        """
        INSERT INTO test_results
            (user_id, url, score, load_time,
             total_links, working_links, broken_links, broken_links_list,
             technologies, trust_score, excitement_score, professionalism_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            user_id,
            result_data.get("url"),
            result_data.get("score"),
            result_data.get("load_time"),
            links.get("total", 0),
            links.get("working", 0),
            links.get("broken", 0),
            json.dumps(links.get("broken_list", [])),
            json.dumps(result_data.get("technologies", [])),
            emotions.get("trust_score", 0),
            emotions.get("excitement_score", 0),
            emotions.get("professionalism_score", 0),
        ),
    )
    test_result_id = cursor.fetchone()["id"]

    cursor.execute(
        "INSERT INTO test_history (user_id, test_result_id, url, score) VALUES (%s, %s, %s, %s)",
        (user_id, test_result_id, result_data.get("url"), result_data.get("score")),
    )

    conn.commit()
    conn.close()
    return test_result_id


def save_ai_suggestion(test_result_id: int, url: str,
                       score: int, broken_count: int, suggestion: str) -> None:
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO ai_suggestions (test_result_id, url, score, broken_count, suggestion)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (test_result_id, url, score, broken_count, suggestion),
    )
    conn.commit()
    conn.close()


def get_user_history(user_id: int) -> list[dict]:
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            th.url,
            th.score,
            th.tested_at,
            tr.load_time,
            tr.total_links,
            tr.working_links,
            tr.broken_links,
            tr.technologies,
            (SELECT suggestion FROM ai_suggestions
             WHERE test_result_id = tr.id
             ORDER BY id DESC LIMIT 1) AS suggestion
        FROM test_history th
        LEFT JOIN test_results tr ON th.test_result_id = tr.id
        WHERE th.user_id = %s
        ORDER BY th.tested_at DESC
        LIMIT 50
        """,
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        try:
            techs = json.loads(row["technologies"] or "[]")
        except (json.JSONDecodeError, TypeError):
            techs = []

        history.append({
            "url":        row["url"],
            "score":      row["score"],
            "tested_at":  row["tested_at"],
            "load_time":  row["load_time"],
            "links": {
                "total":   row["total_links"],
                "working": row["working_links"],
                "broken":  row["broken_links"],
            },
            "technologies": techs,
            "ai_suggestion": row["suggestion"],
        })

    return history


def get_test_detail(test_result_id: int) -> dict | None:
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM test_results WHERE id = %s", (test_result_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_test_full_detail(test_result_id: int) -> dict | None:
    """
    Reconstructs the same 'result' envelope shape that /api/test/website
    returns, from a historical row - so the exact same report
    (PDF/CSV) generation code can be reused for any past test.
    """
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM test_results WHERE id = %s", (test_result_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    cursor.execute(
        "SELECT suggestion FROM ai_suggestions WHERE test_result_id = %s ORDER BY id DESC LIMIT 1",
        (test_result_id,),
    )
    suggestion_row = cursor.fetchone()
    conn.close()

    try:
        broken_list = json.loads(row["broken_links_list"] or "[]")
    except (json.JSONDecodeError, TypeError):
        broken_list = []

    try:
        technologies = json.loads(row["technologies"] or "[]")
    except (json.JSONDecodeError, TypeError):
        technologies = []

    return {
        "test_id": row["id"],
        "url": row["url"],
        "timestamp": row["tested_at"],
        "score": row["score"],
        "load_time": row["load_time"],
        "technologies": technologies,
        "links": {
            "total": row["total_links"],
            "working": row["working_links"],
            "broken": row["broken_links"],
            "broken_list": broken_list,
        },
        "emotions": {
            "trust_score": row["trust_score"],
            "excitement_score": row["excitement_score"],
            "professionalism_score": row["professionalism_score"],
        },
        "ai_suggestion": suggestion_row["suggestion"] if suggestion_row else None,
    }


# ---------------------------------------------------------------------------
# Admin dashboard queries
# ---------------------------------------------------------------------------
def get_all_users() -> list[dict]:
    conn   = get_db()
    cursor = conn.cursor()
    # password_hash is deliberately excluded - never send it to the client
    cursor.execute(
        "SELECT id, username, email, role, created_at FROM users ORDER BY id DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_tests(limit: int = 200) -> list[dict]:
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, user_id, url, score, broken_links, total_links,
               working_links, load_time, tested_at
        FROM test_results
        ORDER BY tested_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_admin_stats() -> dict:
    conn   = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS c FROM users")
    total_users = cursor.fetchone()["c"]

    cursor.execute("SELECT COUNT(*) AS c FROM test_results")
    total_tests = cursor.fetchone()["c"]

    cursor.execute("SELECT COALESCE(AVG(score), 0) AS a FROM test_results")
    avg_score = cursor.fetchone()["a"]

    cursor.execute("SELECT COALESCE(SUM(broken_links), 0) AS b FROM test_results")
    total_broken = cursor.fetchone()["b"]

    # "Successful" test = a test that scored 70 or above
    cursor.execute("SELECT COUNT(*) AS c FROM test_results WHERE score >= 70")
    successful_tests = cursor.fetchone()["c"]
    success_rate = round((successful_tests / total_tests) * 100, 1) if total_tests > 0 else 0

    conn.close()
    return {
        "total_users":   total_users,
        "total_tests":   total_tests,
        "avg_score":     round(avg_score, 1),
        "total_broken":  total_broken,
        "success_rate":  success_rate,
    }


def delete_user(user_id: int) -> None:
    conn   = get_db()
    cursor = conn.cursor()
    # Clean up dependent rows first so we don't leave orphaned records behind
    cursor.execute(
        "DELETE FROM ai_suggestions WHERE test_result_id IN "
        "(SELECT id FROM test_results WHERE user_id = %s)", (user_id,)
    )
    cursor.execute("DELETE FROM test_history WHERE user_id = %s", (user_id,))
    cursor.execute("DELETE FROM test_results WHERE user_id = %s", (user_id,))
    cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
    conn.commit()
    conn.close()


def delete_test(test_id: int) -> None:
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ai_suggestions WHERE test_result_id = %s", (test_id,))
    cursor.execute("DELETE FROM test_history WHERE test_result_id = %s", (test_id,))
    cursor.execute("DELETE FROM test_results WHERE id = %s", (test_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# User dashboard queries
# ---------------------------------------------------------------------------
def get_user_stats(user_id: int) -> dict:
    conn   = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS c FROM test_results WHERE user_id = %s", (user_id,))
    total_tests = cursor.fetchone()["c"]

    cursor.execute("SELECT COALESCE(AVG(score), 0) AS a FROM test_results WHERE user_id = %s", (user_id,))
    avg_score = cursor.fetchone()["a"]

    cursor.execute("SELECT COALESCE(MAX(score), 0) AS m FROM test_results WHERE user_id = %s", (user_id,))
    best_score = cursor.fetchone()["m"]

    cursor.execute("SELECT COALESCE(SUM(broken_links), 0) AS b FROM test_results WHERE user_id = %s", (user_id,))
    total_broken = cursor.fetchone()["b"]

    conn.close()
    return {
        "total_tests":  total_tests,
        "avg_score":    round(avg_score, 1),
        "best_score":   best_score,
        "total_broken": total_broken,
    }


def get_user_tests(user_id: int, limit: int = 20) -> list[dict]:
    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, url, score, broken_links, total_links,
               working_links, load_time, tested_at
        FROM test_results
        WHERE user_id = %s
        ORDER BY tested_at DESC
        LIMIT %s
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]