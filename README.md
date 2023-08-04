# Cabin Pressure Sample

This repo implements an Ansys-Gymnasium interface that can be integrated
easily with [Plato](https://github.com/Azure/plato).

One way to run this sample is to copy all files in the ``src`` folder of this repo
into the ``src`` folder of the
[``getting-started-on-aml``](https://github.com/Azure/plato/tree/main/examples/getting-started-on-aml)
in the Plato repository.

To implement your own simulation, modify these in ``sim.py``:

-   You must provide your own twin model file and define the path in
		``twin_model_file``.
-   Define ``observation_space`` and ``action_space`` as Gymnasium's ``spaces``
-   Define the reward in the ``reward`` method of ``TwinBuilderEnv``
