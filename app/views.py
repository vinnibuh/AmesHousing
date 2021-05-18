# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import base64
from datetime import datetime
from typing import Optional

import redis
import uvicorn
from fastapi import FastAPI, status, HTTPException, Response, Request,BackgroundTasks
from pydantic import BaseModel

from housinglib.func import predict


app = FastAPI()


def decode_none(s):
    if s is None:
        return None
    if isinstance(s, dict):
        return {decode_none(k): decode_none(v) for k, v in s.items()}
    if isinstance(s, list):
        return [decode_none(k) for k in s]
    if isinstance(s, bytes):
        return s.decode(encoding='utf-8')
    return str(s)


def encode_none(s):
    if s is None:
        return None
    if isinstance(s, dict):
        return {encode_none(k): decode_none(v) for k, v in s.items()}
    if isinstance(s, list):
        return [encode_none(k) for k in s]

    return bytes(str(s), encoding='utf-8')


def state_function(func):
    class CustomStr(str):
        def __call__(self):
            return func.__name__, str(datetime.now())

    return CustomStr(func.__name__)


class ModelsModel:
    """
    Models are stored in Redis KV store

    # state markers
    created: str

    # attributes
    name: str
    version: str
    model_data: bytes, should be pickled object
    trained: str
    dataset: str, dataset_id from Redis KV store

    """

    @staticmethod
    def model_hash(model_key):
        return f"models:{model_key}"

    @staticmethod
    def model_key(name, version):
        return f"{name}:{version}"

    @staticmethod
    def link_from_key(model_key):
        model_link = model_key.replace(':', '-')
        return f"models/{model_link}"

    @staticmethod
    def key_from_link(model_link):
        model_key = model_link.replace('-', ':')
        return f"{model_key}"

    @staticmethod
    @state_function
    def created():
        ...

    @staticmethod
    def models():
        return "models"

    @staticmethod
    def predictions(model_hash):
        return f"{model_hash}:predictions"

    @staticmethod
    def from_db(**kwargs):
        # if there is no created key - instance is invalid
        return dict(
            name=decode_none(kwargs.get('name', None)),
            version=decode_none(kwargs.get('version', None)),
            model_data=decode_none(kwargs.get('model_data', None)),
            trained=decode_none(kwargs.get('trained', None)),
            created=decode_none(kwargs['created']),  # will throw
            dataset=decode_none(kwargs.get('dataset', None))
        )

    @staticmethod
    def to_db(**kwargs):
        return dict(
            name=encode_none(kwargs['name']),
            version=encode_none(kwargs['version']),
            model_data=encode_none(kwargs['model_data']),
            trained=encode_none(kwargs['trained']),
            dataset=encode_none(kwargs['dataset'])
            # do not allow state fields to be updated
        )


class AddModel(BaseModel):
    name: str
    version: str
    model_data: str
    trained: str = None
    dataset: str


class GetModel_Out(BaseModel):
    name: str
    version: str
    model_data: str
    trained: str = None
    dataset: str
    created: str


class DatasetsModel:
    """
    Datasets are stored in Redis KV store

    # state markers
    created: str

    # attributes
    name: str
    version: str
    data: bytes, should be pickled object

    """

    @staticmethod
    def dataset_hash(dataset_key):
        return f"datasets:{dataset_key}"

    @staticmethod
    def dataset_key(name, version):
        return f"{name}:{version}"

    @staticmethod
    def link_from_key(dataset_key):
        dataset_link = dataset_key.replace(':', '-')
        return f"datasets/{dataset_link}"

    @staticmethod
    def key_from_link(dataset_link):
        dataset_key = dataset_link.replace('-', ':')
        return f"{dataset_key}"

    @staticmethod
    @state_function
    def created():
        ...

    @staticmethod
    def datasets():
        return "datasets"

    @staticmethod
    def from_db(**kwargs):
        # if there is no created key - instance is invalid
        return dict(
            created=decode_none(kwargs['created']),
            name=decode_none(kwargs.get('name', None)),
            version=decode_none(kwargs.get('version', None)),
            data=decode_none(kwargs.get('data', None))
        )

    @staticmethod
    def to_db(**kwargs):
        return dict(
            name=encode_none(kwargs['name']),
            version=encode_none(kwargs['version']),
            data=encode_none(kwargs['data'])
            # do not allow state fields to be updated
        )


class AddDataset(BaseModel):
    name: str
    version: str
    dataset: bytes


class GetDataset_Out(BaseModel):
    name: str
    version: str
    dataset: bytes
    created: str


class PredictionModel:
    """
    Predictions are stored in Redis KV store

    # state markers
    created: str

    # attributes
    input_values: bytes, should be pickled object
    model_response: bytes, should be pickled object
    user_response: bytes  #TODO

    """

    @staticmethod
    def prediction_hash(model_hash, prediction_id):
        return f"{model_hash}:predictions:{prediction_id}"

    @staticmethod
    def prediction_key():
        return f"{datetime.now().timestamp()}"

    @staticmethod
    def link_from_key(prediction_key):
        prediction_key = prediction_key.replace(':', '-')
        return f"predictions/{prediction_key}"

    @staticmethod
    def key_from_link(prediction_link):
        prediction_key = prediction_link.replace('-', ':')
        return f"{prediction_key}"

    @staticmethod
    @state_function
    def created():
        ...

    @staticmethod
    def from_db(**kwargs):
        return dict(
            input_values=decode_none(kwargs.get('input_values', None)),
            model_response=decode_none(kwargs.get('model_response', None)),
            user_response=decode_none(kwargs.get('user_response', None)),
            created=decode_none(kwargs['created']),  # will throw
        )

    @staticmethod
    def to_db(**kwargs):
        return dict(
            input_values=encode_none(kwargs['input_values']),
            model_response=encode_none(kwargs.get('model_response', None)),
            user_response=encode_none(kwargs.get('user_response', None))
        )


class PostPrediction(BaseModel):
    input_values: bytes
    model_response: bytes
    user_response: str


class RequestsModel:
    """
    Requests are stored in Redis KV store

    # state markers
    created: str

    # attributes
    request_url: str
    request_method: str
    model_id: str, from Redis KV store
    dataset_id: str, from Redis KV store
    input_data: bytes, should be pickled object
    model_response: bytes, should be pickled object
    user_response: bytes  #TODO

    """

    @staticmethod
    def request_hash(request_id):
        return f"requests:{request_id}"

    @staticmethod
    def request_key():
        return f"{datetime.now().timestamp()}"

    @staticmethod
    def link_from_key(request_key):
        request_key = request_key.replace(':', '-')
        return f"requests/{request_key}"

    @staticmethod
    def key_from_link(request_link):
        request_key = request_link.replace('-', ':')
        return f"{request_key}"

    @staticmethod
    @state_function
    def created():
        ...

    @staticmethod
    def requests():
        return f"requests"

    @staticmethod
    def from_db(**kwargs):
        # if there is no created key - instance is invalid
        return dict(
            request_url=decode_none(kwargs.get('request_url', None)),
            request_method=decode_none(kwargs.get('request_method', None)),
            model_id=decode_none(kwargs.get('model_id', None)),
            dataset_id=decode_none(kwargs.get('dataset_id', None)),
            input_data=decode_none(kwargs.get('input_data', None)),
            model_response=decode_none(kwargs.get('model_response', None)),
            user_response=decode_none(kwargs.get('user_response', None)),
            created=decode_none(kwargs['created'])  # will throw
        )

    @staticmethod
    def to_db(**kwargs):
        return dict(
            request_url=encode_none(kwargs['request_url']),
            request_method=encode_none(kwargs['request_method']),
            model_id=encode_none(kwargs.get('model_id', None)),
            dataset_id=encode_none(kwargs.get('dataset_id', None)),
            input_data=encode_none(kwargs.get('input_data', None)),
            model_response=encode_none(kwargs.get('model_response', None)),
            user_response=encode_none(kwargs.get('user_response', None))
            # do not allow state fields to be updated
        )


class GetRequest_Out(BaseModel):
    request_url: str
    request_method: str
    model_id: str = None
    dataset_id: str = None
    input_data: bytes = None
    model_response: bytes = None
    user_response: bytes = None
    created: str


@app.get("/models", status_code=status.HTTP_200_OK)
def get_models_list():
    r = redis.Redis(decode_responses=True)

    result = []
    # get all models sorted by recency
    for p in r.zrevrangebyscore(ModelsModel.models(), "+inf", "-inf", 0, -1):
        result.append(
            dict(
                model_key=decode_none(p),
                url=f"{ModelsModel.link_from_key(decode_none(p))}"
            )
        )

    return result


@app.post("/models", status_code=status.HTTP_201_CREATED)
def add_model(model: AddModel,
              response: Response):
    r = redis.Redis(decode_responses=True)
    model_key = ModelsModel.model_key(model.name, model.version)
    model_hash = ModelsModel.model_hash(model_key)

    with r.pipeline() as pipe:
        try:
            pipe.watch(model_hash)

            res = pipe.hsetnx(model_hash, *ModelsModel.created())
            if res == 1:
                # pipe.multi()
                pipe.hset(model_hash, mapping=ModelsModel.to_db(
                    name=model.name,
                    version=model.version,
                    model_data=model.model_data,
                    trained=model.trained,
                    dataset=model.dataset
                ))

                ts = datetime.now().timestamp()  # store model_key in ordered set
                pipe.zadd(ModelsModel.models(), {model_key: ts})
                # pipe.execute()

                model_url = f"{ModelsModel.link_from_key(model_key)}"
                response.headers['Location'] = model_url
                return

            # if res==0 then hash already exists
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Model already exists"
            )

        except redis.WatchError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Model has just been created by someone else"
            )


@app.get("/models/last", status_code=status.HTTP_200_OK)
def get_last_model():
    r = redis.Redis(decode_responses=True)
    model_id = r.zrevrangebyscore(ModelsModel.models(), "+inf", "-inf", 0, 1)
    if len(model_id) == 0:
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail="There are no models on server"
        )
    model_hash = ModelsModel.model_hash(model_id[0])

    res = r.hgetall(model_hash)
    if len(res) > 0:
        res = ModelsModel.from_db(**decode_none(res))

        return GetModel_Out(**res)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Model is not found"
    )


@app.get("/models/{model_link}", status_code=status.HTTP_200_OK)
def get_model(model_link):
    r = redis.Redis(decode_responses=True)
    model_key = ModelsModel.key_from_link(model_link)
    model_hash = ModelsModel.model_hash(model_key)

    res = r.hgetall(model_hash)
    if len(res) > 0:
        res = ModelsModel.from_db(**decode_none(res))

        return GetModel_Out(**res)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Model is not found"
    )


@app.get("/datasets", status_code=status.HTTP_200_OK)
def get_datasets_list():
    r = redis.Redis(decode_responses=True)

    result = []
    for p in r.smembers(DatasetsModel.datasets()):
        result.append(
            dict(
                model_key=decode_none(p),
                url=f"{DatasetsModel.link_from_key(decode_none(p))}"
            )
        )

    return result


@app.post("/datasets", status_code=status.HTTP_201_CREATED)
def add_dataset(dataset: AddDataset,
                response: Response):
    r = redis.Redis(decode_responses=True)
    dataset_key = DatasetsModel.dataset_key(dataset.name, dataset.version)
    dataset_hash = DatasetsModel.dataset_hash(dataset_key)

    with r.pipeline() as pipe:
        try:
            pipe.watch(dataset_hash)

            res = pipe.hsetnx(dataset_hash, *DatasetsModel.created())
            if res == 1:
                # pipe.multi()
                pipe.hset(dataset_hash, mapping=DatasetsModel.to_db(
                    name=dataset.name,
                    version=dataset.version,
                    dataset=dataset.dataset
                ))
                pipe.sadd(DatasetsModel.datasets(), dataset_key)
                # pipe.execute()

                dataset_url = DatasetsModel.link_from_key(dataset_key)
                response.headers['Location'] = dataset_url
                return

            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset already exists"
            )

        except redis.WatchError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset has just been created by someone else"
            )


@app.get("/datasets/{dataset_link}", status_code=status.HTTP_200_OK)
def get_dataset(dataset_link):
    r = redis.Redis(decode_responses=True)
    dataset_key = DatasetsModel.key_from_link(dataset_link)
    dataset_hash = DatasetsModel.dataset_hash(dataset_key)

    res = r.hgetall(dataset_hash)
    if len(res) > 0:
        res = DatasetsModel.from_db(**decode_none(res))

        return GetDataset_Out(**res)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Dataset is not found"
    )


@app.get("/requests", status_code=status.HTTP_200_OK)
def get_requests(since: Optional[str] = None, to: Optional[str] = None):
    r = redis.Redis(decode_responses=True)
    since_ts = '-inf'
    to_ts = '+inf'
    if since:
        since_ts = datetime.strptime(since, "%Y-%m-%d %H:%M:%S.%f").timestamp()
    if to:
        to_ts = datetime.strptime(to, "%Y-%m-%d %H:%M:%S.%f").timestamp()

    result = []
    for p in r.zrangebyscore(RequestsModel.requests(), since_ts, to_ts):
        result.append(
            dict(
                request_key=decode_none(p),
                url=f"{RequestsModel.link_from_key(decode_none(p))}"
            )
        )

    return result


@app.get("/requests/{request_link}", status_code=status.HTTP_200_OK)
def get_request(request_link):
    r = redis.Redis(decode_responses=True)
    request_key = RequestsModel.key_from_link(request_link)
    request_hash = RequestsModel.request_hash(request_key)

    res = r.hgetall(request_hash)
    if len(res) > 0:
        res = RequestsModel.from_db(**decode_none(res))

        return GetRequest_Out(**res)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Request is not found"
    )


@app.post("/models/{model_link}/predict", status_code=status.HTTP_202_ACCEPTED)
def post_predict(prediction: PostPrediction,
                 model_link: str,
                 request: Request,
                 response: Response,
                 background_tasks: BackgroundTasks):
    r = redis.Redis(decode_responses=True)

    model_key = ModelsModel.key_from_link(model_link)
    model_hash = ModelsModel.model_hash(model_key)
    model_object = r.hgetall(model_hash)
    # safe check model
    if len(model_object) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model is not found"
        )
    if model_object['trained'] is None:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail="This model hasn't been trained yet"
        )

    request_key = RequestsModel.request_key()
    request_hash = RequestsModel.request_hash(request_key)

    prediction_key = PredictionModel.prediction_key()
    prediction_hash = PredictionModel.prediction_hash(model_hash, prediction_key)

    with r.pipeline() as pipe:
        # add entries for request and prediction in Redis KV store
        pipe.hsetnx(request_hash, *RequestsModel.created())
        pipe.hsetnx(prediction_hash, *PredictionModel.created())

        pipe.hset(request_hash, mapping=RequestsModel.to_db(
            request_url=request.url.path,
            request_method=request.method,
            model_id=model_hash,
            input_data=prediction.input_values
        ))
        ts = datetime.now().timestamp()
        pipe.zadd(RequestsModel.requests(), {request_key: ts})

        pipe.hset(prediction_hash, mapping=PredictionModel.to_db(
            input_values=prediction.input_values
        ))
        pipe.zadd(ModelsModel.models(), {model_key: ts})

        pipe.execute()

        # start background process for prediction and return
        background_tasks.add_task(make_predict, prediction_key, model_key, request_key)

        prediction_url = PredictionModel.link_from_key(prediction_key)
        model_url = ModelsModel.link_from_key(model_key)
        response.headers['Location'] = '/'.join([model_url, prediction_url])
        return


async def make_predict(prediction_id, model_id, request_id):
    r = redis.Redis(decode_responses=True)

    model_hash = ModelsModel.model_hash(model_id)  # get model from Redis
    model = r.hget(model_hash, 'model_data')
    model = pickle.loads(model)

    prediction_hash = PredictionModel.prediction_hash(model_hash, prediction_id)  # get data from Redis
    input_values = r.hget(prediction_hash, 'input_values')
    data = pickle.loads(input_values)

    model_output = predict(data, model)  # predict, pickle and store in Redis
    pickled_output = pickle.dumps(model_output)
    r.hset(prediction_hash, "model_response", pickled_output)

    request_hash = RequestsModel.request_hash(request_id)  # update request hash in Redis
    r.hset(request_hash, "model_response", pickled_output)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
