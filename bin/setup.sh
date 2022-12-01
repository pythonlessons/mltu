python3 -m venv venv
activate() {
    . venv/bin/activate
    echo "installing requirements to virtual environment"
    pip install -r requirements.txt
}
activate