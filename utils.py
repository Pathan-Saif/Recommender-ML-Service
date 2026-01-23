# from datetime import datetime

# def map_event_to_weight(event_type: str):
#     mapping = {
#         "view": 0.5,
#         "click": 1.0,
#         "cart": 2.0,
#         "purchase": 3.0
#     }
#     return mapping.get(event_type, 0.5)


def map_event_to_weight(event_type: str):
    mapping = {"view": 0.5, "click": 1.0, "cart": 2.0, "purchase": 3.0}
    return mapping.get(event_type, 0.5)





# PS C:\Users\HP\desktop\ml-service> .\venv\Scripts\activate
# (venv) PS C:\Users\HP\desktop\ml-service> uvicorn app:app --reload