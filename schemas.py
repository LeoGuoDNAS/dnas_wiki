from pydantic import BaseModel, Field

class workorder_status_schema(BaseModel):
    workorder_id: str = Field(
        ..., 
        description="Defines the workorder id of workorder in question"
    )

def statusTranslate(status: str):
    match status:
        case 'current status':
            return 'Translated Status'
        case 'parts ordered':
            return 'Parts Tentatively Arriving'
        case 'waitdispapprvl':
            return 'Tentatively Scheduled'
        case 'scheduled':
            return 'Scheduled'
        case 'parts received': 
            return 'Scheduling'
        case 'waiting quote':
            return 'Generating Quote'
        case 'waiting parts':
            return 'Sourcing Parts'
        case 'waitcustapprvl':
            return 'Quote Needs Approval'
        case 'travel':
            return 'Tech On Route'
        case 'waiting service':
            return 'Tentatively Scheduled'
        case 'checked out':
            return 'Pending Quote Upload'
        case 'on break':
            return 'Parts Returned'
        case 'onsite':
            return 'Service In Progress'
        case 'unacknowledged':
            return 'Scheduling In Progress'
        case 'pending':
            return 'Pending'
        case _:
            return status