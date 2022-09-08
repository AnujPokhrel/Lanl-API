import { gql, useMutation } from '@apollo/client';
import React, { useReducer, useState } from 'react';
import Highlighter from 'react-highlight-words';
import './App.css';

const GET_TAGS = gql`
  mutation GetTags($prThreshold: Float!, $trainingFileName: String!, $text: String!) {
    getTags(prThreshold: $prThreshold, trainingFileName: $trainingFileName, text: $text) {
        tags
        trainingFileName
        prThreshold
        text
    }
  }
`;

const formReducer = (state, event) => {
  if (event.reset) {
    return {
      probability_theshold: 0,
      input_file_name: '',
      input_text: '',
    };
  }
  return {
    ...state,
    [event.name]: event.value,
  };
};


function GetTags() {
    let inputprThreshold;
    let inputtrainingFileName;
    let inputtext;
    let text_to_highlight;
    let fetchedTags = [];
    const [getTags, { data, loading, error }] = useMutation(GET_TAGS);
    const [formData, setFormData] = useReducer(formReducer, {count: 100});
    const [submitting, setSubmitting] = useState(false);
    // const [fetchedTags] = useState(data.getTags);

    // debugger;
    // if (loading) return 'Submitting...';
    // if (loading){
    //   return 'Submitting....';
    // }
    if (error) return `Submission error! ${error.message}`;
    if (data){
      // tt_display = "";
      // parsed_data = JSON.parse(data)
      fetchedTags = data.getTags.tags;
      text_to_highlight = data.getTags.text; 
      // split_text = inputtext.split(' ');
      // split_text.forEach(item=>{
      //   if fetchedTags
      // })
    }

    const handleSubmit = (event) => {
      event.preventDefault();
      setSubmitting(true);
    }

    const handleChange = (event) => {
      const isCheckbox = event.target.type === 'checkbox';

      setFormData({
        prThreshold: event.target.name
      });
    }

    function cleanUp(){
      fetchedTags = [];
      inputprThreshold.value = '';
      inputtrainingFileName.value = '';
      inputtext.value = '';
      debugger;
    }
  
    return (
      <div className='wrapper'>
        <h1>Fetch NER Tags</h1>
        <form
          onSubmit={e => {
            e.preventDefault();
            getTags({ variables: { prThreshold: inputprThreshold.value, trainingFileName: inputtrainingFileName.value, text: inputtext.value } });
            // inputprThreshold.value = '';
            // inputtrainingFileName.value = '';
            // inputtext.value = '';
          }}
        >
          <fieldset disabled={submitting}>
            <label>
              <p>Probability Threshold</p>
              <input
                name='probability_threshold'
                type='number'
                step="0.01"
                min='0'
                max='1'
                onChange={handleChange}
                ref={node => {
                  inputprThreshold = node;
                }}
                disabled={loading}     
              />
            </label>
          </fieldset>
          <fieldset disabled={submitting}>
            <label>
              <p>Training File Name</p>
              <input
                name='training_file_name'
                onChange={handleChange}
                ref={node => {
                  inputtrainingFileName = node;
                }}
                disabled={loading}
              />
            </label>
          </fieldset>
          <fieldset disabled={submitting}>
            <label>
              <p>Input Text</p>
              <textarea
                name='input_text'
                class='big-field'
                onChange={handleChange}
                ref={node => {
                  inputtext = node;
                }}
                disabled={loading}
              />
            </label>
          </fieldset>
          <button type="submit" disabled={submitting}>Get Tags</button>
        </form>
        
        {(fetchedTags.length > 0) ?
          <div>
            <h2>Fetched Tags</h2>
            <div>
              <Highlighter
                highlightClassName="Highlight"
                searchWords={fetchedTags}
                autoEscape={true}
                textToHighlight={text_to_highlight}
              />
            </div> 
            {/* <div>
              <ul>
                {Array.from(fetchedTags).forEach(item => {
                  // debugger;
                  return <li>{item}</li>;
                })}
              </ul>
              {fetchedTags}
            </div> */}
          </div> : null
        }
      </div>
    );
  }

  export default GetTags;