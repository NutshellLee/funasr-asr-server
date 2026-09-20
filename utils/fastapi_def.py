from typing import Dict


from pydantic import BaseModel, Field


# 定义请求体
class Item(BaseModel):
    """定义请求数据体

    Args:
        BaseModel (BaseModel): 基础类
    """

    audio_base64: str = Field(..., description="音频base64")


class ItemFile(BaseModel):
    """定义请求数据体

    Args:
        BaseModel (BaseModel): 基础类
    """

    file_base64: str = Field(..., description="音频base64")
    filename: str = Field(..., description="文件名")


# 定义响应体
class ItemResponse(BaseModel):
    """定义放回响应体

    Args:
        BaseModel (BaseModel): 基础基础类
    """

    statusCode: int = Field(..., description="响应状态码")
    message: str = Field(..., description="响应信息")
    response: Dict = Field(..., description="响应信息")
