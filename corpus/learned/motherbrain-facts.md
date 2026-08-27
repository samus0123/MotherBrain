# MotherBrain operating notes

MotherBrain is a sparse mixture-of-experts language model. The largest
configuration is called the mother preset. The mother preset has 1157
trillion total parameters and activates 6.93 trillion parameters per token.

Information is fed to MotherBrain as patches. Every patch that is applied
creates a new sequential version. Version zero is the base checkpoint.
A patch trains a low rank delta while the base weights stay frozen.
Each patch replays older material so the model does not forget.

The serve command exposes MotherBrain over HTTP on port 8000.
The scale command prices a configuration before it is built.
