from app import app

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=80)

if __name__ == '__main__':
    print("yooo")
    context = ('/home/ubuntu/ssl/cert.crt', '/home/ubuntu/ssl/key.key')
    app.run(host='0.0.0.0', port=80, ssl_context=context, threaded=True, debug=True)
