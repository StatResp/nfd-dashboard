import re


def configuration_invalid(metadata, s3) -> bool: 
    error_string = ""
    if float(metadata['window_size']) > 24 or float(metadata['window_size']) < 1:
        error_string += "ERROR: window_size must be >= 1 and <= 24\n"
    r = re.compile('^\d\d\d\d-\d\d-\d\d$')
    if r.match(metadata['start_date']) is None:
        error_string += "ERROR: start date improperly formatted. Must be %Y-%m-%d\n"
    if r.match(metadata['end_date']) is None:
        error_string += "ERROR: end date improperly formatted. Must be %Y-%m-%d\n"
    if 100 >= float(metadata['alpha']) < 0:
        error_string += "ERROR: alpha must be in range [0,100)\n"
    aws_r = re.compile('^s3:\/\/')
    if aws_r.match(s3['s3_staging_directory']) is None:
        error_string += "ERROR: s3 staging directory incorrect"
    if aws_r.match(s3['s3_save_location']) is None:
        error_string += "ERROR: s3 save location incorrect"
    if error_string == "":
        return False
    else:
        print(error_string)
        return True

