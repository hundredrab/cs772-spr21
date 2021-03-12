function shuffleDevs() {
  var ul = document.querySelector('.navbar-right');
  for (var i = ul.children.length; i >= 0; i--) {
      ul.appendChild(ul.children[Math.random() * i | 0]);
  }
}

var app = new Vue({
  el: '#app',
  data: {
    sent: "",
    message: "-",
    error: "",
    status: "",
    ratings: [],
    model: "layer0",

},
  watch: {
    sent: function(){
      this.populateRes()
    },
    model: function(){
      this.populateRes()
    },
  },

  mounted(){
    this.serverURL = 'http://mars.cse.iitb.ac.in:55555/'
    //this.serverURL = 'https://nlpserver.southeastasia.cloudapp.azure.com:80/'
    this.baseURL = this.serverURL
    shuffleDevs();
  },

  computed: {
    query_text: function() {
      return this.sent + ' 123 ' + this.model
    }
  },

  methods: {
      cssStyle: function(alpha) {
        return {
          backgroundColor: `rgba(0, 100, 0, ${alpha}`
        };
      },
      populateRes: _.debounce(function(id){
        shuffleDevs()
        var app = this

        // TODO: Handle all kinds of emptiness and backspacing here
        console.log(app.sent)
        app.ratings = []
        if (!app.sent) {
          app.status = ""
          return
        }
        app.fetched = false
        app.error = ""
        app.status = "Searching"


        let data = new FormData();
        url = this.baseURL
        axios.get(
          url + app.query_text , data,
        ).then(function (response) {
          console.log(response);
          if (response.status==200){
            app.status = ""
            app.ratings = response.data.probabilities
            app.results = response.data.results
            app.search_time = response.data.search_time
            app.query_id = response.data.query_id
            app.image_link = response.data.image_link
            app.query_image_link = response.data.query_image_link
            app.fetched = true

          }

        }).catch(function(error) {
          app.status = "Something went wrong."
          console.log(error)
          console.log(error.response)
          if(error.response){
            console.log(error.response.data)
            console.log(error.response.status)
            console.log(error.response.headers)
            if(error.response.data.message && error.response.data.message['__all__'])
              app.error = error.response.data.message['__all__'][0]
            else if(error.response.data.detail)
              app.error = error.response.data.detail
            else
              app.error = "An error occured"
          }
          else{
            app.error = error
          }
          app.status = ""
        })
      }, 500),

      send: function(attr) {
        let data = new FormData()
        data.append("query", app.query_id)
        data.append("attr",attr)
        axios.post(
          (this.baseURL + "log/"), data,
        ).then(function (response) {
          console.log(response)
        }).catch(function(error) {
          console.log(error)
        })
      },

      clearAll: function() {
        this.search_box = ""
        this.image_hash = ""
        this.image_link = ""
        this.query_image_link = ""
        this.fact_checked = ""
        this.source = ""
        this.search_text = ""
        this.file = ""

        this.image_url = ""
        this.article_url = ""
        this.query_id = ""
        this.search_time = ""
        this.status = ""
        this.results = ""
        this.hash_tracked = true
      },

      loadTestCase: function(test) {
        let arr = ['search_box', 'image_hash', 'fact_checked', 'source', 'file', 'image_url']
        this.clearAll()
        for (item of arr){
          if (item in this.test_cases[test])
            this[item] = this.test_cases[test][item]
        }
      },
  },
})
