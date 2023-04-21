# ParaFLAI

## Running

to run sequential cifar
```
python basicfl/cifar.py
```

to run server and 2 clients
Terminal 1
```
python server.py --ncli 2
```

Terminal 2
```
python client.py --ncli 2 --id 0
```

Terminal 3
```
python client.py --ncli 2 --id 1
```

## Sources

Code for this project was based around/referenced from Flower (flower.dev) documentation and implementation examples.