package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"time"

	"github.com/blackjack/webcam"
	"github.com/kelseyhightower/envconfig"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Configuration holds the environment variables
type Configuration struct {
	ModelFile  string `required:"true"`
	LabelsFile string `required:"true"`
}

func main() {
	var config Configuration
	err := envconfig.Process("SMARCCTV", &config)
	if err != nil {
		log.Fatal(err)
	}
	// Init tensorflow's graph

	// Load the serialized GraphDef from a file.
	model, err := ioutil.ReadFile(config.ModelFile)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	sessionTF, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer sessionTF.Close()

	// Found the best match. Read the string from labelsFile, which
	// contains one line per label.
	file, err := os.Open(config.LabelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", config.LabelsFile, err)
	}

	// ****************************
	// Init webcam
	type frameSizes []webcam.FrameSize
	//sess := session.Must(session.NewSession())
	//uploader := s3manager.NewUploader(sess)
	cam, err := webcam.Open("/dev/video0") // Open webcam
	if err != nil {
		panic(err.Error())
	}
	defer cam.Close()
	// ...
	// Setup webcam image format and frame size here (see examples or documentation)
	// ...

	formatDesc := cam.GetSupportedFormats()
	var formats []webcam.PixelFormat
	for f := range formatDesc {
		formats = append(formats, f)
	}

	format := formats[1]
	log.Println(format)

	frames := frameSizes(cam.GetSupportedFrameSizes(format))

	size := frames[2]

	f, w, h, err := cam.SetImageFormat(format, uint32(size.MaxWidth), uint32(size.MaxHeight))

	if err != nil {
		panic(err.Error())
	} else {
		fmt.Fprintf(os.Stderr, "Resulting image format: %s (%dx%d)\n", formatDesc[f], w, h)
	}

	err = cam.StartStreaming()
	if err != nil {
		panic(err.Error())
	}
	timeout := uint32(5) //5 seconds
	tick := time.Tick(1000 * time.Millisecond)

	for {
		err = cam.WaitForFrame(timeout)

		switch err.(type) {
		case nil:
		case *webcam.Timeout:
			fmt.Fprint(os.Stderr, err.Error())
			continue
		default:
			panic(err.Error())
		}

		frame, err := cam.ReadFrame()
		if len(frame) != 0 {

			select {
			case <-tick:

				// Run inference on *imageFile.
				// For multiple images, session.Run() can be called in a loop (and
				// concurrently). Alternatively, images can be batched since the model
				// accepts batches of image data as input.
				// decode jpeg into image.Image

				tensor, err := makeTensorFromImage(frame)
				//  tensor, err := makeTensorFromImage(*imagefile)
				if err != nil {
					log.Fatal(err)
				}
				output, err := sessionTF.Run(
					map[tf.Output]*tf.Tensor{
						graph.Operation("Mul").Output(0): tensor,
					},
					[]tf.Output{
						graph.Operation("final_result").Output(0),
					},
					nil)
				if err != nil {
					log.Fatal("==> error ", err)
				}
				// output[0].Value() is a vector containing probabilities of
				// labels for each image in the "batch". The batch size was 1.
				// Find the most probably label index.
				probabilities := output[0].Value().([][]float32)[0]
				//printBestLabel(probabilities, config.LabelsFile)
				if isPeople(probabilities, labels) > 80 {
					fmt.Println("I have seen someone")
				}
			default:
			}
		} else if err != nil {
			log.Println(err)
		}
	}
}
func isPeople(probabilities []float32, labels []string) float32 {
	bestIdx := 0
	for i, p := range probabilities {
		if p > probabilities[bestIdx] {
			bestIdx = i
		}
	}
	if labels[bestIdx] == "people" {
		return probabilities[bestIdx] * 100.0
	}
	return 0
}

func printBestLabel(probabilities []float32, labelsFile string) {
	bestIdx := 0
	for i, p := range probabilities {
		if p > probabilities[bestIdx] {
			bestIdx = i
		}
	}
	// Found the best match. Read the string from labelsFile, which
	// contains one line per label.
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}
	fmt.Printf("BEST MATCH: (%2.0f%% likely) %s\n", probabilities[bestIdx]*100.0, labels[bestIdx])
}

// Convert the image in filename to a Tensor suitable as input to the Inception model.
func makeTensorFromImage(img []byte) (*tf.Tensor, error) {

	//func makeTensorFromImage(filename string) (*tf.Tensor, error) {
	/*
		//func makeTensorFromImage(img []byte) (*tf.Tensor, error) {
		bytes, err := ioutil.ReadFile(filename)
		if err != nil {
			return nil, err
		}
	*/ // DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(img))
	//tensor, err := tf.NewTensor(img)
	if err != nil {
		return nil, err
	}
	// Construct a graph to normalize the image
	graph, input, output, err := constructGraphToNormalizeImage()
	if err != nil {
		return nil, err
	}
	// Execute that graph to normalize this one image
	sessionTF, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer sessionTF.Close()
	normalized, err := sessionTF.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

// The inception model takes as input the image described by a Tensor in a very
// specific normalized format (a particular image size, shape of the input tensor,
// normalized pixel values etc.).
//
// This function constructs a graph of TensorFlow operations which takes as
// input a JPEG-encoded string and returns a tensor suitable as input to the
// inception model.
func constructGraphToNormalizeImage() (graph *tf.Graph, input, output tf.Output, err error) {
	// Some constants specific to the pre-trained model at:
	//
	// - The model was trained after with images scaled to 299x299 pixels.
	// - The colors, represented as R, G, B in 1-byte each were converted to
	//   float using (value - Mean)/Scale.
	const (
		H, W  = 299, 299
		Mean  = float32(0)
		Scale = float32(255)
	)
	// - input is a String-Tensor, where the string the JPEG-encoded image.
	// - The inception model takes a 4D tensor of shape
	//   [BatchSize, Height, Width, Colors=3], where each pixel is
	//   represented as a triplet of floats
	// - Apply normalization on each pixel and use ExpandDims to make
	//   this single image be a "batch" of size 1 for ResizeBilinear.
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s,
						op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()
	return graph, input, output, err
}
