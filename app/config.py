import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-docvqa"

FIELDS = {
    "invoice_number": "What is the invoice number?",
    "invoice_date": "What is the invoice date?",
    "due_date": "What is the payment due date?",
    "bill_to_name": "Who is the invoice billed to?",
    "bill_to_address": "What is the billing address?",
    "total_amount": "What is the total amount payable?",
    "total_tax": "What is the total tax amount?",
    "description": "product Name or service name?"
}


