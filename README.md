# Email-Spam-Classification with REST API

The Email-Spam-Classification project focuses on classifying SMS messages as either "ham" (legitimate) or "spam" using various machine learning models. The dataset used for this project is the SMS Spam Collection, which consists of 5,574 tagged messages in English.

## Content

The dataset contains one message per line, with each line composed of two columns:
- `v1`: the label (either "ham" or "spam")
- `v2`: the raw text of the message

### Data Sources

The dataset has been compiled from various free or research-oriented sources on the Internet:

1. **Grumbletext Web Site**:
   - A collection of 425 SMS spam messages manually extracted from a UK forum where cell phone users publicly report SMS spam messages.
   - Identifying the text of spam messages in these claims was a meticulous and time-consuming task.

2. **NUS SMS Corpus (NSC)**:
   - A subset of 3,375 randomly chosen ham messages from a larger dataset of approximately 10,000 legitimate messages.
   - Collected for research at the Department of Computer Science, National University of Singapore.
   - Messages mostly originate from Singaporeans, especially students at the University, who volunteered their messages knowing they would be made publicly available.

3. **SMS Spam Corpus v.0.1 Big**:
   - Contains 1,002 ham messages and 322 spam messages.
   - This corpus has been utilized in various academic researches.

## Classification Model

The Extra Tree Classifier was specifically used for spam classification, which demonstrated the highest accuracy among the models tested.

## Running the Server

To run the server, follow these steps:

1. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

2. Start the server using Uvicorn:
    ```sh
    uvicorn main:app --reload
    ```

3. The server will start running locally, and you can access the API at `http://localhost:8000`.

## REST API

The REST API provides endpoints for interacting with the spam classification model. Here are the available endpoints:

- `POST /classify`: Endpoint for classifying a single message. Send a JSON object with the message text, and it will return the predicted label ("ham" or "spam").

Example Request:
```json
{
  "message": "Congratulations! You've won a free trip. Click here to claim your prize."
}
```

Output
```json
{
  "label": "spam"
}
```
