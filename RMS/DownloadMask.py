""" Download a mask from the server if a new mask exists.

This section can be added to the .config file in the Calibration section

; Mask
; ----
; Permit remote mask download
mask_download_permissive: true
; Name of the new platepar file on the server
mask_remote_name: mask_latest.bmp
; Name of the directory on the server which contains masks
remote_mask_dir: masks




"""


from __future__ import print_function, division, absolute_import

import datetime
import logging
import os
from os.path import exists as file_exists

import paramiko


from RMS.UploadManager import _agentAuth




# Get the logger from the main module
log = logging.getLogger("logger")




def downloadNewMask(config, port=22):
    """ Connect to the central server and download a new mask file, if available. """

    if config.mask_download_permissive:
        log.info("Mask download enabled")
    else:
        log.info("Mask download disabled")
        return False


    log.info('Checking for new mask on the server...')
    if file_exists(config.rsa_private_key) is False:
        log.debug("Can't contact the server: RSA private key file not found.")
        return False

    log.debug('Establishing SSH connection to: ' + config.hostname + ':' + str(port) + '...')

    try:
        # Connect to host
        t = paramiko.Transport((config.hostname, port))
        t.start_client()

        # Authenticate the connection
        auth_status = _agentAuth(t, config.stationID.lower(), config.rsa_private_key)
        if not auth_status:
            return False

        # Open new SFTP connection
        sftp = paramiko.SFTPClient.from_transport(t)

    except:
        log.error('Connecting to server failed!')
        return False


    # Check that the remote directory exists
    try:
        sftp.stat(config.remote_dir)

    except Exception as e:
        log.error("Remote directory '" + config.remote_dir + "' does not exist!")
        return False


    # Construct path to remote mask directory
    remote_mask_path = config.remote_dir + '/' + config.remote_mask_dir + '/'

    # Change the directory into file
    remote_mask = remote_mask_path + config.mask_remote_name


    # Check if the remote mask file exists
    try:
        sftp.lstat(remote_mask)
    
    except IOError as e:
        log.info('No new mask on the server!')
        return False


    # Download the remote mask
    sftp.get(remote_mask, os.path.join(config.config_file_path, config.mask))

    log.info('Latest mask downloaded!')


    ### Rename the remote mask file, add a timestamp of download
    ### This prevents the same mask being downloaded and overwriting an operators more recent changes

    # Construct a new name with the time of the download included
    dl_mask_name = remote_mask_path + 'mask_dl_' \
        + datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S.%f') + '.cal'

    sftp.posix_rename(remote_mask, dl_mask_name)

    log.info('Remote mask renamed to: ' + dl_mask_name)

    ### ###

    return True




if __name__ == "__main__":

    from RMS.Logger import initLogging
    import RMS.ConfigReader as cr

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Download an updated mask, and rename the remote copy. \
        """, formatter_class=argparse.RawTextHelpFormatter)


    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one.")

    cml_args = arg_parser.parse_args()

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Init the logger
    initLogging(config)


    # Test mask downloading
    downloadNewMask(config, port=22)
