from detection import Detection
from ontology import CaptionOntology

def main():
    # Create a detection object with the caption ontology
    # propmpt:class
    det = Detection(ontology=CaptionOntology({"label on a package":"label"}))
    det.predict(input="image")

if __name__ == "__main__":
    main()
