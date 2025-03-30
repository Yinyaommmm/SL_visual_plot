import os
from typing import Literal
def export_result(plt,purename: str,extension:Literal['pdf','png'] = 'pdf'):
    plt.savefig(f'./image/{purename}.{extension}',format=extension,bbox_inches='tight')